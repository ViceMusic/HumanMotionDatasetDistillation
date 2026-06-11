import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from backbones.simlpe import SiMLPeMotionBackbone, build_simlpe_config, expmap_to_simlpe_xyz66


@dataclass
class DistillConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"
    synthetic_len: int = 100 # 合成的新时序的长度，必须 >= input_len + output_len
    feature_dim: int = 99    # 数据的特征维度，H36M的expmap是99维的，backbone内部会自己调整，不用管
    top_k: int = 16          # 取得前6个主频成分进行匹配，过多可能会引入噪声，过少可能无法捕捉动作特征，16是一个经验值，可以调整看看效果
    batch_size: int = 8      # 一次用八个窗口
    outer_steps: int = 100   # 训练的总步数，越多合成的时序可能越好，但也越慢，100是一个初始值，可以根据需要调整
    num_backbones: int = 3   # 每个蒸馏step在几个随机初始化的backbone附近匹配梯度
    backbone_reinit_interval: int = 20  # 每隔多少个step重新随机初始化一组backbone
    lr_synthetic: float = 1e-2
    # 权重系数，可以根据实际情况调整，看看哪个损失对最终效果影响更大，或者是否需要引入更多的损失项（比如骨骼长度保持等物理约束）
    lambda_harm: float = 1.0
    lambda_grad: float = 1.0
    lambda_rigid: float = 0.0
    lambda_vel: float = 1.0
    lambda_pred: float = 1.0
    window_mode: str = "random"
    seed: int = 888
    save_name: str = "h36m_expmap_sequences_distilled.npz" # 合成后的时序数据保存文件名

# 确保输出目录存在，如果不存在则创建
def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

# 将可能是bytes的值转换为字符串，方便后续处理
def as_text(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    elif value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


class SyntheticMotionBank(nn.Module):
    """One learnable expmap sub-series [100, 99] for each action category."""

    def __init__(self, action_names, synthetic_len=100, feature_dim=99, init_std=0.02):
        super().__init__()
        self.action_names = list(action_names)
        self.action_to_idx = {name: idx for idx, name in enumerate(self.action_names)}
        bank = torch.randn(len(self.action_names), synthetic_len, feature_dim) * init_std
        self.synthetic_motions = nn.Parameter(bank)

    def forward(self, actions):
        if isinstance(actions, torch.Tensor):
            action_ids = actions.to(self.synthetic_motions.device)
        else:
            action_ids = torch.tensor([self.action_to_idx[str(a)] for a in actions], device=self.synthetic_motions.device)
        return self.synthetic_motions[action_ids]

    def get_all(self):
        return {
            "action_names": self.action_names,
            "synthetic_motions": self.synthetic_motions.detach().cpu(),
        }


class RealSubseriesSampler:
    """Samples action-balanced real expmap sub-series [B, 100, 99] from NPZ."""

    def __init__(self, data_path, synthetic_len=100, train_subjects_only=True):
        self.synthetic_len = synthetic_len
        self.by_action = {}
        data = np.load(data_path, allow_pickle=True)
        for subject, action, motion in zip(data["subjects"], data["actions"], data["motions"]):
            subject_name = as_text(subject)
            if train_subjects_only and subject_name in ("S5", "5"):
                continue
            motion = np.asarray(motion, dtype=np.float32)
            if motion.ndim != 2 or motion.shape[1] != 99 or motion.shape[0] < synthetic_len:
                continue
            self.by_action.setdefault(as_text(action).lower(), []).append(motion)

        self.action_names = sorted(self.by_action.keys())
        if not self.action_names:
            raise ValueError("No valid sequences with length >= {} found in {}".format(synthetic_len, data_path))

    def sample(self, batch_size, actions=None):
        if actions is None:
            actions = [self.action_names[i % len(self.action_names)] for i in range(batch_size)]
            random.shuffle(actions)

        subs = []
        for action in actions:
            seq = random.choice(self.by_action[str(action)])
            start = random.randint(0, seq.shape[0] - self.synthetic_len)
            subs.append(seq[start : start + self.synthetic_len])
        return torch.tensor(np.stack(subs), dtype=torch.float32), actions

# =======================================================================
# 它把真实动作序列和合成动作序列都做 rFFT，
# real_sub.shape      # [B, M, 99]
# synthetic_sub.shape # [B, M, 99]
# 找出真实动作里最主要的若干个频率成分，
# 然后要求 synthetic sequence 在这些主要频率上的振幅和真实序列接近。
# =======================================================================
def compute_harmonic_loss(real_sub, synthetic_sub, top_k=16, p=2):
    """Match rFFT amplitudes on real-data dominant harmonics.

    real_sub/synthetic_sub are expmap tensors shaped [B, M, 99].
    """
    f_real = torch.fft.rfft(real_sub, dim=1)
    f_syn = torch.fft.rfft(synthetic_sub, dim=1)
    amp_real = torch.abs(f_real)
    amp_syn = torch.abs(f_syn)
    score = amp_real.detach().mean(dim=(0, 2))
    harmonic_idx = torch.topk(score, k=min(top_k, score.numel())).indices
    diff = amp_real[:, harmonic_idx, :] - amp_syn[:, harmonic_idx, :]
    loss = diff.abs().mean() if p == 1 else diff.pow(2).mean()
    return loss, harmonic_idx

# =======================================================================
# get_harmonic_reconstruction 本质上就是一个频域降噪 / 主成分提取函数，
# 它不是梯度匹配的核心公式本身。
# ======================================================================
def get_harmonic_reconstruction(real_sub, synthetic_sub, top_k=16):
    """Reconstruct expmap sequences using real-data top-k rFFT harmonics."""
    seq_len = real_sub.shape[1]
    f_real = torch.fft.rfft(real_sub, dim=1)
    f_syn = torch.fft.rfft(synthetic_sub, dim=1)
    score = torch.abs(f_real).detach().mean(dim=(0, 2))
    harmonic_idx = torch.topk(score, k=min(top_k, score.numel())).indices

    real_filtered = torch.zeros_like(f_real)
    syn_filtered = torch.zeros_like(f_syn)
    real_filtered[:, harmonic_idx, :] = f_real[:, harmonic_idx, :]
    syn_filtered[:, harmonic_idx, :] = f_syn[:, harmonic_idx, :]
    return (
        torch.fft.irfft(real_filtered, n=seq_len, dim=1),
        torch.fft.irfft(syn_filtered, n=seq_len, dim=1),
        harmonic_idx,
    )

# =======================================================================
# 从一段长度为 100 的 motion sub-series 里切出姿态预测训练样本。
# 例如 input_len=10, output_len=25 就是用前10帧预测后25帧。
# 这里的窗口切法有两种，"first" 就是直接从开头切，"random" 就是在 [0, 100-10-25] 的范围内随机切。
# =======================================================================
def make_windows_from_subseries(series, input_len, output_len, num_windows=None, window_mode="random"):
    """Create expmap prediction windows using externally supplied siMLPe lengths."""
    batch_size, synthetic_len, _ = series.shape
    total_len = input_len + output_len
    if synthetic_len < total_len:
        raise ValueError("synthetic_len={} is shorter than input_len + output_len={}".format(synthetic_len, total_len))

    windows_per_series = 1 if num_windows is None else num_windows
    max_start = synthetic_len - total_len
    past_windows = []
    future_windows = []
    for batch_idx in range(batch_size):
        for _ in range(windows_per_series):
            if window_mode == "first":
                start = 0
            elif window_mode == "random":
                start = torch.randint(0, max_start + 1, (1,), device=series.device).item()
            else:
                raise ValueError("Unknown window_mode: {}".format(window_mode))
            window = series[batch_idx, start : start + total_len]
            past_windows.append(window[:input_len])
            future_windows.append(window[input_len:total_len])
    return torch.stack(past_windows, dim=0), torch.stack(future_windows, dim=0)

# =======================================================================
# 计算预测损失和速度损失，都是 MSE 形式的。
# 这里的 pred_xyz66 是 backbone 的输出，gt_future_expmap 是 ground truth 的 expmap，需要先转换成 xyz66 再计算损失。
# =======================================================================
def prediction_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)
    return F.mse_loss(pred_xyz66, gt_xyz66)


def velocity_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)
    pred_vel = pred_xyz66[:, 1:] - pred_xyz66[:, :-1]
    gt_vel = gt_xyz66[:, 1:] - gt_xyz66[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)

#=======================================================================
# 计算刚性损失，这里暂时返回0，因为我们没有具体的刚性约束实现。这个函数是为了后续扩展用的，如果你想加入一些骨骼长度保持或者其他物理约束，可以在这里实现。
#=======================================================================
def compute_rigid_loss(reference_tensor):
    return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)

# ======================================================================
# 这一步就是匹配一步梯度
# 主要是匹配这两个玩意
# 用真实数据训练 backbone 时产生的参数梯度
# ≈
# 用合成数据训练 backbone 时产生的参数梯度
# =======================================================================
def compute_gradient_matching_loss(backbone, real_past, real_future, syn_past, syn_future, output_len, eps=1e-8):
    """One-step approximation of HDT/MTT trajectory matching.

    Real gradients are the target update direction. Synthetic gradients keep graph
    connectivity so SyntheticMotionBank can be optimized.
    """
    params = [p for p in backbone.parameters() if p.requires_grad]

    real_pred = backbone(real_past, output_len=output_len)
    real_loss = prediction_loss(real_pred, real_future) + velocity_loss(real_pred, real_future)
    g_real = torch.autograd.grad(real_loss, params, allow_unused=True)
    g_real = [None if grad is None else grad.detach() for grad in g_real]

    syn_pred = backbone(syn_past, output_len=output_len)
    syn_loss = prediction_loss(syn_pred, syn_future) + velocity_loss(syn_pred, syn_future)
    g_syn = torch.autograd.grad(syn_loss, params, create_graph=True, allow_unused=True)

    numerator = torch.zeros((), device=syn_past.device)
    denominator = torch.zeros((), device=syn_past.device)
    for real_grad, syn_grad in zip(g_real, g_syn):
        if real_grad is None or syn_grad is None:
            continue
        numerator = numerator + (syn_grad - real_grad).pow(2).sum()
        denominator = denominator + real_grad.pow(2).sum()
    return numerator / (denominator + eps)


def build_random_backbones(config, num_backbones, device):
    backbones = []
    for _ in range(num_backbones):
        backbone = SiMLPeMotionBackbone(config).to(device)
        backbone.train()
        for param in backbone.parameters():
            param.requires_grad_(True)
        backbones.append(backbone)
    return backbones

# 这里从 SyntheticMotionBank 里取出所有 synthetic data。
def save_synthetic_bank_npz(path, bank, step, logs, feature_type="expmap", feature_dim=99):
    """Save distilled motions with the same NPZ schema as processed H36M data."""
    ensure_dir(os.path.dirname(path))
    payload = bank.get_all()
    action_names = payload["action_names"]
    synthetic_motions = payload["synthetic_motions"].numpy().astype(np.float32)

    subjects = np.array(["synthetic"] * len(action_names), dtype=object)
    actions = np.array(action_names, dtype=object)
    trials = np.array([1] * len(action_names), dtype=object)
    lengths = np.array([motion.shape[0] for motion in synthetic_motions], dtype=np.int64)
    raw_paths = np.array(
        ["distilled://step_{}/{}_1".format(step, action) for action in action_names],
        dtype=object,
    )
    motions = np.empty(len(action_names), dtype=object)
    for idx, motion in enumerate(synthetic_motions):
        motions[idx] = motion

    np.savez(
        path,
        subjects=subjects,
        actions=actions,
        trials=trials,
        lengths=lengths,
        raw_paths=raw_paths,
        motions=motions,
        feature_type=np.array(feature_type, dtype=object),
        feature_dim=np.array(feature_dim, dtype=np.int64),
        distill_step=np.array(step, dtype=np.int64),
        distill_logs=np.array(logs, dtype=object),
    )


def train_distillation():
    cfg = DistillConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 设置backbone和预测输入输出（这里使用siMLPe作为基础内容）
    simlpe_config = build_simlpe_config()
    input_len = simlpe_config.motion.h36m_input_length
    output_len = simlpe_config.motion.h36m_target_length_train

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = RealSubseriesSampler(cfg.data_path, synthetic_len=cfg.synthetic_len)
    # 内置一个synthetic_motions: [num_actions, 100, 99]
    bank = SyntheticMotionBank(sampler.action_names, cfg.synthetic_len, cfg.feature_dim).to(device)
    # 多个随机初始化backbone；只用于产生梯度匹配目标，不被optimizer更新。
    backbones = build_random_backbones(simlpe_config, cfg.num_backbones, device)

    # 优化器和目录什么的
    optimizer = torch.optim.Adam(bank.parameters(), lr=cfg.lr_synthetic)
    output_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(os.path.dirname(__file__), cfg.output_dir)
    save_path = os.path.join(output_dir, cfg.save_name)

    logs = []
    for step in range(1, cfg.outer_steps + 1): # 这个相当于iter
        if cfg.backbone_reinit_interval > 0 and step > 1 and (step - 1) % cfg.backbone_reinit_interval == 0:
            backbones = build_random_backbones(simlpe_config, cfg.num_backbones, device)

        # 这里返回的是：
        # real_sub.shape = [B, 100, 99]
        # 以及actions是类型名称
        real_sub, actions = sampler.sample(cfg.batch_size)
        real_sub = real_sub.to(device)
        # 根据action获取同类别的合成数据片段，synthetic_sub.shape = [B, 100, 99]
        syn_sub = bank(actions)

        # real_sub -> FFT -> 真实频谱 vs syn_sub  -> FFT -> 合成频谱
        l_harm, harmonic_idx = compute_harmonic_loss(real_sub, syn_sub, top_k=cfg.top_k)
        # 顺便算了一下降噪版本：只保留主要趋势/周期后的动作片段
        real_h, syn_h, _ = get_harmonic_reconstruction(real_sub, syn_sub, top_k=cfg.top_k)

        # 切割窗口
        # real_past:   [N, 50, 99]  real_future: [N, 10, 99]
        # syn_past:    [N, 50, 99]  syn_future:  [N, 10, 99]
        real_past, real_future = make_windows_from_subseries(real_h, input_len, output_len, window_mode=cfg.window_mode)
        syn_past, syn_future = make_windows_from_subseries(syn_h, input_len, output_len, window_mode=cfg.window_mode)

        # 计算梯度匹配损失，预测损失，速度损失，刚性损失（目前是0），然后加权求和得到总损失
        grad_losses = [
            compute_gradient_matching_loss(backbone, real_past, real_future, syn_past, syn_future, output_len)
            for backbone in backbones
        ]
        l_grad = torch.stack(grad_losses).mean()
        syn_pred = backbones[0](syn_past, output_len=output_len)
        l_pred_syn = prediction_loss(syn_pred, syn_future) # 这两个是原有的内容，让合成数据约束的合理一点
        l_vel_syn = velocity_loss(syn_pred, syn_future)
        l_rigid = compute_rigid_loss(syn_sub) # 这个暂时是0，如果以后有了有具体的刚性约束实现，可以在 compute_rigid_loss 里实现。

        total_loss = (
            cfg.lambda_harm * l_harm
            + cfg.lambda_grad * l_grad
            + cfg.lambda_rigid * l_rigid
            + cfg.lambda_vel * l_vel_syn
            + cfg.lambda_pred * l_pred_syn
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "L_harm": float(l_harm.detach().cpu()),
            "L_grad": float(l_grad.detach().cpu()),
            "L_pred_syn": float(l_pred_syn.detach().cpu()),
            "L_vel_syn": float(l_vel_syn.detach().cpu()),
            "L_rigid": float(l_rigid.detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
            "harmonics": harmonic_idx.detach().cpu().tolist(),
        }
        logs.append(row)
        print(
            "step {step} L_harm={L_harm:.6f} L_grad={L_grad:.6f} "
            "L_pred_syn={L_pred_syn:.6f} L_vel_syn={L_vel_syn:.6f} "
            "L_rigid={L_rigid:.6f} L_total={L_total:.6f}".format(**row)
        )

    save_synthetic_bank_npz(save_path, bank, cfg.outer_steps, logs, feature_dim=cfg.feature_dim)
    print("Saved synthetic motion bank to {}".format(save_path))


if __name__ == "__main__":
    train_distillation()
