import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from backbones.simlpe import SiMLPeMotionBackbone, build_simlpe_config, expmap_to_simlpe_xyz66


'''
1 backbone
TM
'''

DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]


@dataclass
class DistillConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"

    synthetic_len: int = 100
    feature_dim: int = 99
    top_k: int = 16

    batch_size: int = 64
    outer_steps: int = 8000

    # 单一 backbone
    num_backbones: int = 1

    # 不重置 backbone
    backbone_reinit_interval: int = 0

    lr_synthetic: float = 1e-2

    lambda_harm: float = 0
    lambda_grad: float = 0.1
    lambda_rigid: float = 0.0
    lambda_vel: float = 0
    lambda_pred: float = 0

    # trajectory matching 内循环参数
    inner_lr: float = 1e-3
    real_inner_steps: int = 5
    syn_inner_steps: int = 5

    window_mode: str = "random"
    seed: int = 888

    save_name: str = "h36m_expmap_sequences_distilled_bs=64_iter=8000_1backbone_no_reinit_traj"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def as_text(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    elif value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def normalize_subject(value):
    text = as_text(value)
    return text if text.startswith("S") else "S{}".format(text)


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
            action_ids = torch.tensor(
                [self.action_to_idx[str(a)] for a in actions],
                device=self.synthetic_motions.device,
            )
        return self.synthetic_motions[action_ids]

    def get_all(self):
        return {
            "action_names": self.action_names,
            "synthetic_motions": self.synthetic_motions.detach().cpu(),
        }


class RealSubseriesSampler:
    """Samples action-balanced real expmap sub-series [B, 100, 99] from NPZ."""

    def __init__(self, data_path, synthetic_len=100, include_subjects=None, heldout_subjects=None):
        self.synthetic_len = synthetic_len
        self.by_action = {}
        self.data = np.load(data_path, allow_pickle=True)
        self.heldout_entries = []

        include_subjects = set(include_subjects or [])
        heldout_subjects = set(heldout_subjects or [])

        for idx, (subject, action, motion) in enumerate(
            zip(self.data["subjects"], self.data["actions"], self.data["motions"])
        ):
            subject_name = normalize_subject(subject)

            if subject_name in heldout_subjects:
                self.heldout_entries.append(idx)
                continue

            if include_subjects and subject_name not in include_subjects:
                continue

            motion = np.asarray(motion, dtype=np.float32)

            if motion.ndim != 2 or motion.shape[1] != 99 or motion.shape[0] < synthetic_len:
                continue

            self.by_action.setdefault(as_text(action).lower(), []).append(motion)

        self.action_names = sorted(self.by_action.keys())

        if not self.action_names:
            raise ValueError(
                "No valid sequences with length >= {} found in {}".format(
                    synthetic_len,
                    data_path,
                )
            )

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

    def get_heldout_entries(self):
        entries = []

        for idx in self.heldout_entries:
            motion = np.asarray(self.data["motions"][idx], dtype=np.float32)
            entries.append(
                {
                    "subject": self.data["subjects"][idx],
                    "action": self.data["actions"][idx],
                    "trial": self.data["trials"][idx],
                    "length": motion.shape[0],
                    "raw_path": self.data["raw_paths"][idx],
                    "motion": motion,
                }
            )

        return entries


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


def make_windows_from_subseries(series, input_len, output_len, num_windows=None, window_mode="random"):
    """Create expmap prediction windows using externally supplied siMLPe lengths."""
    batch_size, synthetic_len, _ = series.shape
    total_len = input_len + output_len

    if synthetic_len < total_len:
        raise ValueError(
            "synthetic_len={} is shorter than input_len + output_len={}".format(
                synthetic_len,
                total_len,
            )
        )

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


def prediction_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)
    return F.mse_loss(pred_xyz66, gt_xyz66)


def velocity_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)

    pred_vel = pred_xyz66[:, 1:] - pred_xyz66[:, :-1]
    gt_vel = gt_xyz66[:, 1:] - gt_xyz66[:, :-1]

    return F.mse_loss(pred_vel, gt_vel)


def compute_rigid_loss(reference_tensor):
    return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)


def get_named_trainable_params(model):
    return {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def get_named_buffers(model):
    return {
        name: b
        for name, b in model.named_buffers()
    }


def model_forward_with_params(model, params, buffers, past, output_len):
    state = {}
    state.update(params)
    state.update(buffers)

    return functional_call(
        model,
        state,
        (past,),
        {"output_len": output_len},
    )


def inner_update(
    model,
    params,
    buffers,
    past,
    future,
    output_len,
    inner_lr,
    create_graph,
):
    pred = model_forward_with_params(
        model,
        params,
        buffers,
        past,
        output_len,
    )

    loss = prediction_loss(pred, future) + velocity_loss(pred, future)

    param_names = list(params.keys())
    param_values = [params[name] for name in param_names]

    grads = torch.autograd.grad(
        loss,
        param_values,
        create_graph=create_graph,
        allow_unused=True,
    )

    new_params = {}

    for name, param, grad in zip(param_names, param_values, grads):
        if grad is None:
            new_params[name] = param
        else:
            new_params[name] = param - inner_lr * grad

    return new_params, loss


def compute_gradient_matching_loss(
    backbone,
    real_past,
    real_future,
    syn_past,
    syn_future,
    output_len,
    inner_lr=1e-3,
    real_steps=5,
    syn_steps=5,
    eps=1e-8,
):
    """
    Trajectory matching version.

    theta_real = T_i(theta, real_h)
    theta_syn  = T_j(theta, syn_h)

    L_grad = ||theta_syn - theta_real||^2 / ||theta_0 - theta_real||^2
    """
    base_params = get_named_trainable_params(backbone)
    buffers = get_named_buffers(backbone)

    real_params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in base_params.items()
    }

    syn_params = {
        name: p.detach().clone().requires_grad_(True)
        for name, p in base_params.items()
    }

    for _ in range(real_steps):
        real_params, _ = inner_update(
            backbone,
            real_params,
            buffers,
            real_past,
            real_future,
            output_len,
            inner_lr,
            create_graph=False,
        )

        real_params = {
            name: p.detach().requires_grad_(True)
            for name, p in real_params.items()
        }

    for _ in range(syn_steps):
        syn_params, _ = inner_update(
            backbone,
            syn_params,
            buffers,
            syn_past,
            syn_future,
            output_len,
            inner_lr,
            create_graph=True,
        )

    numerator = torch.zeros((), device=syn_past.device)
    denominator = torch.zeros((), device=syn_past.device)

    for name in base_params.keys():
        theta_0 = base_params[name].detach()
        theta_real = real_params[name].detach()
        theta_syn = syn_params[name]

        numerator = numerator + (theta_syn - theta_real).pow(2).sum()
        denominator = denominator + (theta_0 - theta_real).pow(2).sum()

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


def save_synthetic_bank_npz(
    path,
    bank,
    step,
    logs,
    distill_subjects,
    heldout_entries,
    feature_type="expmap",
    feature_dim=99,
):
    """Save distilled motions with the same NPZ schema as processed H36M data."""
    ensure_dir(os.path.dirname(path))

    payload = bank.get_all()
    action_names = payload["action_names"]
    synthetic_motions = payload["synthetic_motions"].numpy().astype(np.float32)

    subjects = []
    actions = []
    trials = []
    lengths = []
    raw_paths = []
    motions = []

    for subject in distill_subjects:
        for action, motion in zip(action_names, synthetic_motions):
            subjects.append(subject)
            actions.append(action)
            trials.append(1)
            lengths.append(motion.shape[0])
            raw_paths.append("distilled://step_{}/{}/{}_1".format(step, subject, action))
            motions.append(motion)

    for entry in heldout_entries:
        subjects.append(entry["subject"])
        actions.append(entry["action"])
        trials.append(entry["trial"])
        lengths.append(entry["length"])
        raw_paths.append(entry["raw_path"])
        motions.append(entry["motion"])

    motion_array = np.empty(len(motions), dtype=object)

    for idx, motion in enumerate(motions):
        motion_array[idx] = motion

    np.savez(
        path,
        subjects=np.array(subjects, dtype=object),
        actions=np.array(actions, dtype=object),
        trials=np.array(trials, dtype=object),
        lengths=np.array(lengths, dtype=np.int64),
        raw_paths=np.array(raw_paths, dtype=object),
        motions=motion_array,
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

    simlpe_config = build_simlpe_config()
    input_len = simlpe_config.motion.h36m_input_length
    output_len = simlpe_config.motion.h36m_target_length_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = RealSubseriesSampler(
        cfg.data_path,
        synthetic_len=cfg.synthetic_len,
        include_subjects=DISTILL_SUBJECTS,
        heldout_subjects=HELDOUT_SUBJECTS,
    )

    bank = SyntheticMotionBank(
        sampler.action_names,
        cfg.synthetic_len,
        cfg.feature_dim,
    ).to(device)

    backbones = build_random_backbones(
        simlpe_config,
        cfg.num_backbones,
        device,
    )

    optimizer = torch.optim.Adam(
        bank.parameters(),
        lr=cfg.lr_synthetic,
    )

    output_dir = (
        cfg.output_dir
        if os.path.isabs(cfg.output_dir)
        else os.path.join(os.path.dirname(__file__), cfg.output_dir)
    )
    save_path = os.path.join(output_dir, cfg.save_name)

    logs = []

    print("device:", device)
    print("num_backbones:", cfg.num_backbones)
    print("backbone_reinit_interval:", cfg.backbone_reinit_interval)
    print("inner_lr:", cfg.inner_lr)
    print("real_inner_steps:", cfg.real_inner_steps)
    print("syn_inner_steps:", cfg.syn_inner_steps)
    print("save_path:", save_path)

    for step in range(1, cfg.outer_steps + 1):
        if (
            cfg.backbone_reinit_interval > 0
            and step > 1
            and (step - 1) % cfg.backbone_reinit_interval == 0
        ):
            backbones = build_random_backbones(
                simlpe_config,
                cfg.num_backbones,
                device,
            )

        real_sub, actions = sampler.sample(cfg.batch_size)
        real_sub = real_sub.to(device)

        syn_sub = bank(actions)

        l_harm, harmonic_idx = compute_harmonic_loss(
            real_sub,
            syn_sub,
            top_k=cfg.top_k,
        )

        real_h, syn_h, _ = get_harmonic_reconstruction(
            real_sub,
            syn_sub,
            top_k=cfg.top_k,
        )

        real_past, real_future = make_windows_from_subseries(
            real_h,
            input_len,
            output_len,
            window_mode=cfg.window_mode,
        )

        syn_past, syn_future = make_windows_from_subseries(
            syn_h,
            input_len,
            output_len,
            window_mode=cfg.window_mode,
        )

        grad_losses = [
            compute_gradient_matching_loss(
                backbone,
                real_past,
                real_future,
                syn_past,
                syn_future,
                output_len,
                inner_lr=cfg.inner_lr,
                real_steps=cfg.real_inner_steps,
                syn_steps=cfg.syn_inner_steps,
            )
            for backbone in backbones
        ]

        l_grad = torch.stack(grad_losses).mean()

        syn_pred = backbones[0](
            syn_past,
            output_len=output_len,
        )
        l_pred_syn = prediction_loss(
            syn_pred,
            syn_future,
        )
        l_vel_syn = velocity_loss(
            syn_pred,
            syn_future,
        )
        l_rigid = compute_rigid_loss(
            syn_sub,
        )

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
            "L_harm": float((cfg.lambda_harm * l_harm).detach().cpu()),
            "L_grad": float((cfg.lambda_grad * l_grad).detach().cpu()),
            "L_pred_syn": float((cfg.lambda_pred * l_pred_syn).detach().cpu()),
            "L_vel_syn": float((cfg.lambda_vel * l_vel_syn).detach().cpu()),
            "L_rigid": float((cfg.lambda_rigid * l_rigid).detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
            "harmonics": harmonic_idx.detach().cpu().tolist(),
        }

        logs.append(row)

        if step % 10 == 0:
            print(
                "step {step} L_harm={L_harm:.6f} L_grad={L_grad:.6f} "
                "L_pred_syn={L_pred_syn:.6f} L_vel_syn={L_vel_syn:.6f} "
                "L_rigid={L_rigid:.6f} L_total={L_total:.6f}".format(**row)
            )

    save_synthetic_bank_npz(
        save_path,
        bank,
        cfg.outer_steps,
        logs,
        distill_subjects=DISTILL_SUBJECTS,
        heldout_entries=sampler.get_heldout_entries(),
        feature_dim=cfg.feature_dim,
    )

    print("Saved synthetic motion bank to {}".format(save_path))


if __name__ == "__main__":
    train_distillation()