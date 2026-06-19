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

from backbones.simlpe import (
    SiMLPeMotionBackbone,
    build_simlpe_config,
    expmap_to_simlpe_xyz66,
)


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

    num_backbones: int = 3
    backbone_reinit_interval: int = 20

    lr_synthetic: float = 1e-2

    # Paper-style HDT objective:
    # total_loss = lambda_grad * L_grad + lambda_harm * L_harm
    lambda_harm: float = 0.01
    lambda_grad: float = 1.0

    # These are kept only for logging compatibility; not used in total_loss.
    lambda_rigid: float = 0.0
    lambda_vel: float = 0.0
    lambda_pred: float = 0.0

    # "grad" is cheaper one-step gradient matching.
    # "trajectory" is closer to paper Eq.12 / MTT-style multi-step matching.
    matching_mode: str = "grad"

    # Used only when matching_mode == "trajectory".
    inner_lr: float = 1e-3
    real_inner_steps: int = 5
    syn_inner_steps: int = 5

    window_mode: str = "random"
    seed: int = 888

    # Whether top-k harmonics are selected per channel.
    # For H36M expmap 99-dim, channelwise=True is safer than one global top-k.
    channelwise_harmonics: bool = True

    # Whether to exclude DC component when choosing top-k harmonics.
    # False is closer to paper range [0, floor(M/2)].
    exclude_dc: bool = False

    print_interval: int = 10

    save_name: str = "h36m_expmap_sequences_distilled_hdt_freq_bank_bs64_iter8000.npz"


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


class FrequencySyntheticMotionBank(nn.Module):
    """
    Paper-style frequency-domain synthetic motion bank.

    We initialize a time-domain synthetic motion S, convert it to F_S = rFFT(S),
    and optimize F_S directly. Forward returns iFFT(F_S), so downstream code
    still sees normal motion sequences shaped [B, M, C].
    """

    def __init__(self, action_names, synthetic_len=100, feature_dim=99, init_std=0.02):
        super().__init__()
        self.action_names = list(action_names)
        self.action_to_idx = {name: idx for idx, name in enumerate(self.action_names)}
        self.synthetic_len = synthetic_len
        self.feature_dim = feature_dim

        init_motion = torch.randn(len(self.action_names), synthetic_len, feature_dim) * init_std
        init_freq = torch.fft.rfft(init_motion, dim=1)

        self.freq_real = nn.Parameter(init_freq.real)
        self.freq_imag = nn.Parameter(init_freq.imag)

    def get_freq(self):
        return torch.complex(self.freq_real, self.freq_imag)

    def get_time(self):
        return torch.fft.irfft(self.get_freq(), n=self.synthetic_len, dim=1)

    def forward(self, actions):
        all_motion = self.get_time()

        if isinstance(actions, torch.Tensor):
            action_ids = actions.to(all_motion.device)
        else:
            action_ids = torch.tensor(
                [self.action_to_idx[str(a)] for a in actions],
                device=all_motion.device,
                dtype=torch.long,
            )

        return all_motion[action_ids]

    @torch.no_grad()
    def project_valid_rfft(self):
        """
        For real-valued iFFT, DC and Nyquist imaginary parts should be zero.
        torch.irfft mostly ignores invalid parts, but explicitly zeroing them
        avoids useless parameters drifting.
        """
        self.freq_imag[:, 0, :].zero_()
        if self.synthetic_len % 2 == 0:
            self.freq_imag[:, -1, :].zero_()

    def get_all(self):
        return {
            "action_names": self.action_names,
            "synthetic_motions": self.get_time().detach().cpu(),
        }


class RealSubseriesSampler:
    """Samples action-balanced real expmap sub-series [B, M, 99] from NPZ."""

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

            if motion.ndim != 2:
                continue
            if motion.shape[1] != 99:
                continue
            if motion.shape[0] < synthetic_len:
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


def build_global_harmonic_mask(real_sub, top_k=16, exclude_dc=False):
    """
    real_sub: [B, M, C]
    return mask: [1, M_fft, 1]
    """
    f_real = torch.fft.rfft(real_sub, dim=1)
    score = torch.abs(f_real).detach().mean(dim=(0, 2))  # [M_fft]

    if exclude_dc and score.numel() > 1:
        score = score.clone()
        score[0] = -float("inf")

    k = min(top_k, score.numel())
    idx = torch.topk(score, k=k, dim=0).indices

    mask = torch.zeros(score.shape[0], dtype=torch.bool, device=real_sub.device)
    mask[idx] = True

    return mask.view(1, -1, 1)


def build_channelwise_harmonic_mask(real_sub, top_k=16, exclude_dc=False):
    """
    real_sub: [B, M, C]
    return mask: [1, M_fft, C]
    """
    f_real = torch.fft.rfft(real_sub, dim=1)
    score = torch.abs(f_real).detach().mean(dim=0)  # [M_fft, C]

    if exclude_dc and score.shape[0] > 1:
        score = score.clone()
        score[0, :] = -float("inf")

    m_fft, c = score.shape
    k = min(top_k, m_fft)

    idx = torch.topk(score, k=k, dim=0).indices  # [k, C]

    mask = torch.zeros_like(score, dtype=torch.bool)  # [M_fft, C]
    channel_ids = torch.arange(c, device=real_sub.device).view(1, c).expand(k, c)
    mask[idx, channel_ids] = True

    return mask.unsqueeze(0)


def harmonic_filter_and_loss(
    real_sub,
    syn_sub,
    top_k=16,
    p=2,
    channelwise=True,
    exclude_dc=False,
):
    """
    Paper-style harmonic step.

    real_sub: [B, M, C]
    syn_sub:  [B, M, C]

    1. F_X = FFT(X_sub), F_S = FFT(S)
    2. Select top-k harmonics from real frequency amplitudes.
    3. Build filtered F_X_tilde and F_S_tilde.
    4. L_harm = || |F_X_tilde| - |F_S_tilde| ||_p
    5. X_H = iFFT(F_X_tilde), S_H = iFFT(F_S_tilde)
    """
    seq_len = real_sub.shape[1]

    f_real = torch.fft.rfft(real_sub, dim=1)
    f_syn = torch.fft.rfft(syn_sub, dim=1)

    if channelwise:
        mask = build_channelwise_harmonic_mask(
            real_sub,
            top_k=top_k,
            exclude_dc=exclude_dc,
        )
    else:
        mask = build_global_harmonic_mask(
            real_sub,
            top_k=top_k,
            exclude_dc=exclude_dc,
        )

    f_real_h = torch.zeros_like(f_real)
    f_syn_h = torch.zeros_like(f_syn)

    f_real_h = torch.where(mask, f_real, f_real_h)
    f_syn_h = torch.where(mask, f_syn, f_syn_h)

    amp_diff = torch.abs(f_real_h.detach()) - torch.abs(f_syn_h)

    if p == 1:
        l_harm = amp_diff.abs().mean()
    else:
        l_harm = amp_diff.pow(2).mean()

    real_h = torch.fft.irfft(f_real_h, n=seq_len, dim=1)
    syn_h = torch.fft.irfft(f_syn_h, n=seq_len, dim=1)

    return l_harm, real_h, syn_h, mask


def make_windows_from_subseries(series, input_len, output_len, num_windows=None, window_mode="random"):
    """
    Create expmap prediction windows.

    series: [B, synthetic_len, C]
    return:
        past:   [B * windows_per_series, input_len, C]
        future: [B * windows_per_series, output_len, C]
    """
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


def compute_prediction_objective(backbone, past, future, output_len):
    pred = backbone(past, output_len=output_len)
    return prediction_loss(pred, future) + velocity_loss(pred, future)


def compute_gradient_matching_loss(
    backbone,
    real_past,
    real_future,
    syn_past,
    syn_future,
    output_len,
    eps=1e-8,
):
    """
    Cheaper HDT-lite version:
    one-step gradient matching on harmonic-reconstructed real/synthetic signals.

    This is not full Eq.12, but it is much cheaper and often useful as a smoke test.
    """
    params = [p for p in backbone.parameters() if p.requires_grad]

    real_loss = compute_prediction_objective(
        backbone,
        real_past,
        real_future,
        output_len,
    )
    g_real = torch.autograd.grad(
        real_loss,
        params,
        allow_unused=True,
    )
    g_real = [None if grad is None else grad.detach() for grad in g_real]

    syn_loss = compute_prediction_objective(
        backbone,
        syn_past,
        syn_future,
        output_len,
    )
    g_syn = torch.autograd.grad(
        syn_loss,
        params,
        create_graph=True,
        allow_unused=True,
    )

    numerator = torch.zeros((), device=syn_past.device)
    denominator = torch.zeros((), device=syn_past.device)

    for real_grad, syn_grad in zip(g_real, g_syn):
        if real_grad is None or syn_grad is None:
            continue
        numerator = numerator + (syn_grad - real_grad).pow(2).sum()
        denominator = denominator + real_grad.pow(2).sum()

    return numerator / (denominator + eps)


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


def compute_trajectory_matching_loss(
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
    Closer to paper Eq.12:

        L_grad = || T_j(theta, S_H) - T_i(theta, X_H) ||^2
                 /
                 || theta - T_i(theta, X_H) ||^2

    This is more expensive than one-step gradient matching.
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
        theta0 = base_params[name].detach()
        theta_real = real_params[name].detach()
        theta_syn = syn_params[name]

        numerator = numerator + (theta_syn - theta_real).pow(2).sum()
        denominator = denominator + (theta0 - theta_real).pow(2).sum()

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
    """
    Save distilled motions with the same NPZ schema as processed H36M data.

    Since bank is frequency-domain, bank.get_all() returns iFFT(F_S), i.e. final S.
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simlpe_config = build_simlpe_config()
    input_len = simlpe_config.motion.h36m_input_length
    output_len = simlpe_config.motion.h36m_target_length_train

    sampler = RealSubseriesSampler(
        cfg.data_path,
        synthetic_len=cfg.synthetic_len,
        include_subjects=DISTILL_SUBJECTS,
        heldout_subjects=HELDOUT_SUBJECTS,
    )

    bank = FrequencySyntheticMotionBank(
        sampler.action_names,
        synthetic_len=cfg.synthetic_len,
        feature_dim=cfg.feature_dim,
    ).to(device)

    bank.project_valid_rfft()

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

    print("Device:", device)
    print("Actions:", sampler.action_names)
    print("Input length:", input_len, "Output length:", output_len)
    print("Matching mode:", cfg.matching_mode)
    print("Save path:", save_path)

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

        l_harm, real_h, syn_h, harmonic_mask = harmonic_filter_and_loss(
            real_sub,
            syn_sub,
            top_k=cfg.top_k,
            p=2,
            channelwise=cfg.channelwise_harmonics,
            exclude_dc=cfg.exclude_dc,
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

        if cfg.matching_mode == "trajectory":
            grad_losses = [
                compute_trajectory_matching_loss(
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
        elif cfg.matching_mode == "grad":
            grad_losses = [
                compute_gradient_matching_loss(
                    backbone,
                    real_past,
                    real_future,
                    syn_past,
                    syn_future,
                    output_len,
                )
                for backbone in backbones
            ]
        else:
            raise ValueError("Unknown matching_mode: {}".format(cfg.matching_mode))

        l_grad = torch.stack(grad_losses).mean()

        # Not used in paper-style HDT objective. Kept as zeros for logging compatibility.
        l_pred_syn = torch.zeros((), device=device)
        l_vel_syn = torch.zeros((), device=device)
        l_rigid = compute_rigid_loss(syn_sub)

        total_loss = (
            cfg.lambda_harm * l_harm
            + cfg.lambda_grad * l_grad
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bank.project_valid_rfft()

        with torch.no_grad():
            syn_time = bank.get_time()
            syn_mean = float(syn_time.mean().detach().cpu())
            syn_std = float(syn_time.std().detach().cpu())
            syn_min = float(syn_time.min().detach().cpu())
            syn_max = float(syn_time.max().detach().cpu())

        row = {
            "step": step,
            "L_harm": float((cfg.lambda_harm * l_harm).detach().cpu()),
            "L_grad": float((cfg.lambda_grad * l_grad).detach().cpu()),
            "L_pred_syn": float(l_pred_syn.detach().cpu()),
            "L_vel_syn": float(l_vel_syn.detach().cpu()),
            "L_rigid": float(l_rigid.detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
            "syn_mean": syn_mean,
            "syn_std": syn_std,
            "syn_min": syn_min,
            "syn_max": syn_max,
            "harmonic_count": int(harmonic_mask.sum().detach().cpu()),
        }

        logs.append(row)

        if step % cfg.print_interval == 0:
            print(
                "step {step} "
                "L_harm={L_harm:.6f} "
                "L_grad={L_grad:.6f} "
                "L_total={L_total:.6f} "
                "syn_mean={syn_mean:.6f} "
                "syn_std={syn_std:.6f} "
                "syn_min={syn_min:.6f} "
                "syn_max={syn_max:.6f} "
                "harmonic_count={harmonic_count}".format(**row)
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