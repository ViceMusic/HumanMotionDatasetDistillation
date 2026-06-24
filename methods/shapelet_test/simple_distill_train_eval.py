import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from backbones.simlpe import (
    SiMLPeMotionBackbone,
    build_simlpe_config,
    expmap_to_simlpe_xyz66,
    expmap_to_xyz32,
)


# ============================================================
# Sequence-level parallel HDT-style motion distillation + train
# Save as:
#   methods/smoke_test/run_sequence_hdt_distill_and_train.py
#
# Run:
#   python methods/smoke_test/run_sequence_hdt_distill_and_train.py
#
# Core rule:
#   One real sequence has one synthetic sequence.
#   loss(real_seq_i, syn_seq_i) only updates syn_seq_i.
#   No action-level averaging, no subject-action bank.
# ============================================================


DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]
TEST_SUBJECTS = ["S5"]


@dataclass
class DistillConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"

    synthetic_len: int = 100
    feature_dim: int = 99
    top_k: int = 16
    use_hdt_filter: bool = False  # False: use raw sequence; True: use harmonic-filtered sequence

    batch_size: int = 16
    outer_steps: int = 8000

    num_backbones: int = 1 # 先调整为1试一试
    backbone_reinit_interval: int = 0  # 0 means fixed random backbones, no reinitialization during distillation

    lr_synthetic: float = 1e-2

    # fixed prior
    lambda_harm: float = 0 # 暂时为0
    lambda_grad: float = 0.1
    lambda_vel: float = 300.0
    lambda_pred: float = 300.0

    window_mode: str = "random"
    seed: int = 888
    print_interval: int = 10

    save_name: str = "h36m_expmap_sequences_distilled_sequence_hdt_h001_g01_bs64_iter8000.npz"


@dataclass
class TrainConfig:
    train_npz_path: str = ""
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"

    log_path: str = "logs/train_sequence_hdt_h001_g01_bs64_iter8000.txt"

    train_subjects = DISTILL_SUBJECTS
    test_subjects = TEST_SUBJECTS

    sample_rate: int = 1
    batch_size: int = 64
    num_workers: int = 4
    total_iters: int = 8000
    print_every: int = 100
    seed: int = 888
    lr: float = 3e-4
    weight_decay: float = 1e-4


RESULT_KEYS = ["#2", "#4", "#8", "#10", "#14", "#18", "#22", "#25"]
JOINT_USED_XYZ = np.array(
    [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
).astype(np.int64)
JOINT_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
JOINT_EQUAL = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)


def root_dir():
    return os.path.dirname(__file__)


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(root_dir(), path)


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


def make_seq_key(idx, subject, action, trial, raw_path):
    subject = normalize_subject(subject)
    action = as_text(action).lower()
    trial = as_text(trial)
    raw_path = as_text(raw_path)
    return "{}::{}::{}::{}".format(idx, subject, action, trial), subject, action, trial, raw_path


class SequenceFrequencySyntheticMotionBank(nn.Module):
    """
    Frequency-domain synthetic bank indexed by original training sequence id.

    Row k corresponds to one real sequence, not one action and not one subject-action pair.
    """

    def __init__(self, seq_infos, synthetic_len=100, feature_dim=99, init_motion=None, init_std=0.02):
        super().__init__()
        self.seq_infos = list(seq_infos)
        self.keys = [info["key"] for info in self.seq_infos]
        self.key_to_idx = {key: idx for idx, key in enumerate(self.keys)}
        self.synthetic_len = synthetic_len
        self.feature_dim = feature_dim

        if init_motion is None:
            init_motion = torch.randn(len(self.keys), synthetic_len, feature_dim) * init_std
        else:
            init_motion = torch.as_tensor(init_motion, dtype=torch.float32)
            expected_shape = (len(self.keys), synthetic_len, feature_dim)
            if tuple(init_motion.shape) != expected_shape:
                raise ValueError(
                    "init_motion shape {} does not match expected {}".format(
                        tuple(init_motion.shape),
                        expected_shape,
                    )
                )

        init_freq = torch.fft.rfft(init_motion, dim=1)

        self.freq_real = nn.Parameter(init_freq.real)
        self.freq_imag = nn.Parameter(init_freq.imag)

    def get_freq(self):
        return torch.complex(self.freq_real, self.freq_imag)

    def get_time(self):
        return torch.fft.irfft(self.get_freq(), n=self.synthetic_len, dim=1)

    def forward(self, keys):
        all_motion = self.get_time()
        ids = torch.tensor(
            [self.key_to_idx[str(key)] for key in keys],
            device=all_motion.device,
            dtype=torch.long,
        )
        return all_motion[ids]

    @torch.no_grad()
    def project_valid_rfft(self):
        self.freq_imag[:, 0, :].zero_()
        if self.synthetic_len % 2 == 0:
            self.freq_imag[:, -1, :].zero_()

    def get_all(self):
        return {
            "seq_infos": self.seq_infos,
            "synthetic_motions": self.get_time().detach().cpu(),
        }


class SequenceRealSubseriesSampler:
    """
    Samples from individual real sequences.
    Each key maps to exactly one original sequence.
    """

    def __init__(self, data_path, synthetic_len=100, include_subjects=None, heldout_subjects=None):
        self.synthetic_len = synthetic_len
        self.data = np.load(data_path, allow_pickle=True)
        self.by_key = {}
        self.seq_infos = []
        self.heldout_entries = []

        include_subjects = set(include_subjects or [])
        heldout_subjects = set(heldout_subjects or [])

        zipped = zip(
            self.data["subjects"],
            self.data["actions"],
            self.data["trials"],
            self.data["raw_paths"],
            self.data["motions"],
        )

        for idx, (subject, action, trial, raw_path, motion) in enumerate(zipped):
            subject_name = normalize_subject(subject)
            motion = np.asarray(motion, dtype=np.float32)

            if subject_name in heldout_subjects:
                self.heldout_entries.append(idx)
                continue

            if include_subjects and subject_name not in include_subjects:
                continue

            if motion.ndim != 2 or motion.shape[1] != 99:
                continue
            if motion.shape[0] < synthetic_len:
                continue

            key, subject_name, action_name, trial_name, raw_path_text = make_seq_key(
                idx, subject_name, action, trial, raw_path
            )
            self.by_key[key] = motion
            self.seq_infos.append(
                {
                    "key": key,
                    "subject": subject_name,
                    "action": action_name,
                    "trial": trial_name,
                    "raw_path": raw_path_text,
                    "original_length": motion.shape[0],
                }
            )

        self.keys = [info["key"] for info in self.seq_infos]

        if not self.keys:
            raise ValueError("No valid training sequences with length >= {} found in {}".format(synthetic_len, data_path))

    def sample(self, batch_size, keys=None):
        if keys is None:
            keys = random.sample(self.keys, k=min(batch_size, len(self.keys)))
            if len(keys) < batch_size:
                keys = keys + random.choices(self.keys, k=batch_size - len(keys))

        subs = []
        for key in keys:
            seq = self.by_key[str(key)]
            start = random.randint(0, seq.shape[0] - self.synthetic_len)
            subs.append(seq[start : start + self.synthetic_len])

        return torch.tensor(np.stack(subs), dtype=torch.float32), keys

    def get_initial_subseries(self):
        """
        Initialize each synthetic sequence with one real crop from the same original sequence.
        The order strictly follows self.seq_infos, so it matches SequenceFrequencySyntheticMotionBank rows.
        """
        subs = []
        for info in self.seq_infos:
            key = info["key"]
            seq = self.by_key[str(key)]
            start = random.randint(0, seq.shape[0] - self.synthetic_len)
            subs.append(seq[start : start + self.synthetic_len])
        return torch.tensor(np.stack(subs), dtype=torch.float32)

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


def build_channelwise_harmonic_mask(real_sub, top_k=16):
    f_real = torch.fft.rfft(real_sub, dim=1)
    score = torch.abs(f_real).detach().mean(dim=0)
    m_fft, c = score.shape
    k = min(top_k, m_fft)

    idx = torch.topk(score, k=k, dim=0).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    channel_ids = torch.arange(c, device=real_sub.device).view(1, c).expand(k, c)
    mask[idx, channel_ids] = True
    return mask.unsqueeze(0)


def harmonic_filter_and_loss(real_sub, syn_sub, top_k=16):
    seq_len = real_sub.shape[1]

    f_real = torch.fft.rfft(real_sub, dim=1)
    f_syn = torch.fft.rfft(syn_sub, dim=1)

    mask = build_channelwise_harmonic_mask(real_sub, top_k=top_k)

    f_real_h = torch.where(mask, f_real, torch.zeros_like(f_real))
    f_syn_h = torch.where(mask, f_syn, torch.zeros_like(f_syn))

    l_harm = (torch.abs(f_real_h.detach()) - torch.abs(f_syn_h)).pow(2).mean()

    real_h = torch.fft.irfft(f_real_h, n=seq_len, dim=1)
    syn_h = torch.fft.irfft(f_syn_h, n=seq_len, dim=1)

    return l_harm, real_h, syn_h, mask


def make_windows_from_subseries(series, input_len, output_len, num_windows=None, window_mode="random"):
    batch_size, synthetic_len, _ = series.shape
    total_len = input_len + output_len

    if synthetic_len < total_len:
        raise ValueError("synthetic_len={} is shorter than input_len+output_len={}".format(synthetic_len, total_len))

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


def compute_prediction_objective(backbone, past, future, output_len):
    pred = backbone(past, output_len=output_len)
    return prediction_loss(pred, future) + velocity_loss(pred, future)


def compute_gradient_matching_loss_single(backbone, real_past, real_future, syn_past, syn_future, output_len, eps=1e-8):
    """
    Single-sequence gradient matching.

    This function receives windows from one real sequence and one synthetic sequence only.
    No cross-sequence aggregated gradient is used.
    """
    params = [p for p in backbone.parameters() if p.requires_grad]

    real_loss = compute_prediction_objective(backbone, real_past, real_future, output_len)
    g_real = torch.autograd.grad(real_loss, params, allow_unused=True)
    g_real = [None if grad is None else grad.detach() for grad in g_real]

    syn_loss = compute_prediction_objective(backbone, syn_past, syn_future, output_len)
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


def save_sequence_bank_npz(path, bank, step, logs, heldout_entries, feature_type="expmap", feature_dim=99):
    ensure_dir(os.path.dirname(path))

    payload = bank.get_all()
    seq_infos = payload["seq_infos"]
    synthetic_motions = payload["synthetic_motions"].numpy().astype(np.float32)

    subjects = []
    actions = []
    trials = []
    lengths = []
    raw_paths = []
    motions = []

    for info, motion in zip(seq_infos, synthetic_motions):
        subjects.append(info["subject"])
        actions.append(info["action"])
        trials.append(info["trial"])
        lengths.append(motion.shape[0])
        raw_paths.append("distilled://step_{}/{}".format(step, info["key"]))
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

    sampler = SequenceRealSubseriesSampler(
        cfg.data_path,
        synthetic_len=cfg.synthetic_len,
        include_subjects=DISTILL_SUBJECTS,
        heldout_subjects=HELDOUT_SUBJECTS,
    )

    init_motion = sampler.get_initial_subseries()

    bank = SequenceFrequencySyntheticMotionBank(
        sampler.seq_infos,
        synthetic_len=cfg.synthetic_len,
        feature_dim=cfg.feature_dim,
        init_motion=init_motion,
    ).to(device)
    bank.project_valid_rfft()

    backbones = build_random_backbones(simlpe_config, cfg.num_backbones, device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=cfg.lr_synthetic)

    output_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(root_dir(), cfg.output_dir)
    save_path = os.path.join(output_dir, cfg.save_name)

    logs = []

    print("Device:", device)
    print("Num training sequences:", len(sampler.keys))
    print("Input length:", input_len, "Output length:", output_len)
    print("lambda_harm:", cfg.lambda_harm, "lambda_grad:", cfg.lambda_grad)
    print("lambda_vel:", cfg.lambda_vel, "lambda_pred:", cfg.lambda_pred)
    print("Backbone reinit interval:", cfg.backbone_reinit_interval)
    print("Synthetic initialization: real sequence crop")
    print("Save path:", save_path)

    for step in range(1, cfg.outer_steps + 1):
        real_sub_batch, keys = sampler.sample(cfg.batch_size)
        real_sub_batch = real_sub_batch.to(device)
        syn_sub_batch = bank(keys)

        per_seq_losses = []
        per_seq_harms = []
        per_seq_grads = []
        per_seq_preds = []
        per_seq_vels = []
        harmonic_counts = []

        # Strict sequence-level loss:
        # each real_sub_i only compares to syn_sub_i.
        # total_loss is just an average of independent per-sequence losses.
        for i in range(real_sub_batch.shape[0]):
            real_sub = real_sub_batch[i : i + 1]
            syn_sub = syn_sub_batch[i : i + 1]

            if cfg.use_hdt_filter:
                l_harm, real_h, syn_h, harmonic_mask = harmonic_filter_and_loss(
                    real_sub,
                    syn_sub,
                    top_k=cfg.top_k,
                )
                harmonic_count = int(harmonic_mask.sum().detach().cpu())
            else:
                l_harm = torch.zeros((), device=device)
                real_h = real_sub
                syn_h = syn_sub
                harmonic_count = 0
            


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
                compute_gradient_matching_loss_single(
                    backbone,
                    real_past,
                    real_future,
                    syn_past,
                    syn_future,
                    output_len,
                )
                for backbone in backbones
            ]
            l_grad = torch.stack(grad_losses).mean()

            syn_pred = backbones[0](syn_past, output_len=output_len)
            l_pred_syn = prediction_loss(syn_pred, syn_future)
            l_vel_syn = velocity_loss(syn_pred, syn_future)

            loss_i = (
                cfg.lambda_harm * l_harm
                + cfg.lambda_grad * l_grad
                + cfg.lambda_pred * l_pred_syn
                + cfg.lambda_vel * l_vel_syn
            )

            per_seq_losses.append(loss_i)
            per_seq_harms.append(cfg.lambda_harm * l_harm.detach())
            per_seq_grads.append(cfg.lambda_grad * l_grad.detach())
            per_seq_preds.append(cfg.lambda_pred * l_pred_syn.detach())
            per_seq_vels.append(cfg.lambda_vel * l_vel_syn.detach())
            harmonic_counts.append(harmonic_count)
            
        total_loss = torch.stack(per_seq_losses).mean()

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
            "L_harm": float(torch.stack(per_seq_harms).mean().detach().cpu()),
            "L_grad": float(torch.stack(per_seq_grads).mean().detach().cpu()),
            "L_pred_syn": float(torch.stack(per_seq_preds).mean().detach().cpu()),
            "L_vel_syn": float(torch.stack(per_seq_vels).mean().detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
            "syn_mean": syn_mean,
            "syn_std": syn_std,
            "syn_min": syn_min,
            "syn_max": syn_max,
            "harmonic_count": int(np.mean(harmonic_counts)),
            "example_key": keys[0],
        }
        logs.append(row)

        if step % cfg.print_interval == 0:
            print(
                "distill step {step} "
                "L_harm={L_harm:.6f} "
                "L_grad={L_grad:.6f} "
                "L_pred_syn={L_pred_syn:.6f} "
                "L_vel_syn={L_vel_syn:.6f} "
                "L_total={L_total:.6f} "
                "syn_mean={syn_mean:.6f} "
                "syn_std={syn_std:.6f} "
                "syn_min={syn_min:.6f} "
                "syn_max={syn_max:.6f} "
                "example_key={example_key}".format(**row)
            )

    save_sequence_bank_npz(
        save_path,
        bank,
        cfg.outer_steps,
        logs,
        heldout_entries=sampler.get_heldout_entries(),
        feature_dim=cfg.feature_dim,
    )

    print("Saved distilled sequence-level synthetic bank to {}".format(save_path))
    return save_path


class H36MExpmapWindowDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=1):
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.seqs = []
        self.data_idx = []
        wanted = set(subjects)

        data = np.load(resolve_path(npz_path), allow_pickle=True)
        seq_idx = 0
        for subject, motion in zip(data["subjects"], data["motions"]):
            if normalize_subject(subject) not in wanted:
                continue
            motion = np.asarray(motion, dtype=np.float32)
            if motion.ndim != 2 or motion.shape[1] != 99:
                continue
            motion = motion[np.arange(0, motion.shape[0], sample_rate)]
            if motion.shape[0] < self.total_len:
                continue
            valid_starts = np.arange(0, motion.shape[0] - self.total_len + 1, shift_step)
            self.seqs.append(torch.from_numpy(motion).float())
            self.data_idx.extend(zip([seq_idx] * len(valid_starts), valid_starts.tolist()))
            seq_idx += 1

        if not self.data_idx:
            raise ValueError("No training windows found in {} for subjects {}".format(npz_path, subjects))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        seq_idx, start = self.data_idx[index]
        window = self.seqs[seq_idx][start : start + self.total_len]
        return window[: self.input_len], window[self.input_len :]


class SequenceWindowBatchSampler(Sampler):
    """
    Yield batches from one sequence at a time.

    If a sequence has 41 windows and batch_size=64, it yields one batch of 41.
    If a sequence has 130 windows and batch_size=64, it yields 64, 64, 2.
    No batch mixes windows from different sequences.
    """

    def __init__(self, dataset, batch_size, shuffle_sequences=True, shuffle_windows=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_windows = shuffle_windows

        self.seq_to_indices = {}
        for global_idx, (seq_idx, _start) in enumerate(dataset.data_idx):
            self.seq_to_indices.setdefault(seq_idx, []).append(global_idx)

        self.seq_ids = list(self.seq_to_indices.keys())

    def __iter__(self):
        seq_ids = self.seq_ids[:]
        if self.shuffle_sequences:
            random.shuffle(seq_ids)

        for seq_id in seq_ids:
            indices = self.seq_to_indices[seq_id][:]
            if self.shuffle_windows:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if batch:
                    yield batch

    def __len__(self):
        total = 0
        for indices in self.seq_to_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


class H36MExpmapEvalDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=1):
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.seqs = []
        self.data_idx = []
        wanted = set(subjects)

        data = np.load(resolve_path(npz_path), allow_pickle=True)
        seq_idx = 0
        for subject, motion in zip(data["subjects"], data["motions"]):
            if normalize_subject(subject) not in wanted:
                continue
            motion = np.asarray(motion, dtype=np.float32)
            if motion.ndim != 2 or motion.shape[1] != 99:
                continue
            motion = motion[np.arange(0, motion.shape[0], sample_rate)]
            if motion.shape[0] < self.total_len:
                continue
            valid_starts = np.arange(0, motion.shape[0] - self.total_len + 1, shift_step)
            self.seqs.append(torch.from_numpy(motion).float())
            self.data_idx.extend(zip([seq_idx] * len(valid_starts), valid_starts.tolist()))
            seq_idx += 1

        if not self.data_idx:
            raise ValueError("No eval windows found in {} for subjects {}".format(npz_path, subjects))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        seq_idx, start = self.data_idx[index]
        window = self.seqs[seq_idx][start : start + self.total_len]
        motion_xyz32 = expmap_to_xyz32(window.unsqueeze(0)).squeeze(0) / 1000.0
        return window[: self.input_len], motion_xyz32[self.input_len :]


def evaluate_mpjpe(model, npz_path, subjects, config):
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_eval
    dataset = H36MExpmapEvalDataset(npz_path, subjects, input_len, output_len)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
    )

    sums = np.zeros([output_len], dtype=np.float64)
    num_samples = 0
    model.eval()

    for motion_input_expmap, motion_target_xyz32 in dataloader:
        motion_input_expmap = motion_input_expmap.cuda()
        motion_target_xyz32 = motion_target_xyz32.float()
        batch_size = motion_input_expmap.shape[0]
        num_samples += batch_size

        outputs = []
        current_input = expmap_to_simlpe_xyz66(motion_input_expmap)
        step = config.motion.h36m_target_length_train
        num_step = 1 if step == output_len else output_len // step + 1

        with torch.no_grad():
            for _ in range(num_step):
                output_xyz66 = model.forward_xyz66(current_input, output_len=step)
                outputs.append(output_xyz66.detach().cpu())
                current_input = torch.cat([current_input[:, step:], output_xyz66], dim=1)

        pred_xyz66 = torch.cat(outputs, dim=1)[:, :output_len]
        pred_rot = pred_xyz66.reshape(batch_size, output_len, 22, 3)
        motion_pred = motion_target_xyz32.clone().reshape(batch_size, output_len, 32, 3)
        motion_gt = motion_target_xyz32.clone().reshape(batch_size, output_len, 32, 3)
        motion_pred[:, :, JOINT_USED_XYZ] = pred_rot
        motion_pred[:, :, JOINT_TO_IGNORE] = motion_pred[:, :, JOINT_EQUAL]

        mpjpe = torch.sum(torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim=3), dim=2), dim=0)
        sums += mpjpe.cpu().numpy()

    mpjpe_by_frame = sums / num_samples
    ret = {}
    for idx in range(output_len):
        ret["#{}".format(idx + 1)] = mpjpe_by_frame[idx]
    return [round(ret[key], 1) for key in RESULT_KEYS]


def train_step(model, past_expmap, future_expmap, optimizer, output_len):
    pred_xyz66 = model(past_expmap.cuda(), output_len=output_len)
    gt_xyz66 = expmap_to_simlpe_xyz66(future_expmap.cuda())
    loss_pred = F.mse_loss(pred_xyz66, gt_xyz66)
    loss_vel = velocity_loss(pred_xyz66, future_expmap.cuda())
    loss = loss_pred + loss_vel

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_pred.item(), loss_vel.item()


def train_and_evaluate(train_npz_path):
    cfg = TrainConfig()
    cfg.train_npz_path = train_npz_path

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    config = build_simlpe_config()
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_train

    model = SiMLPeMotionBackbone(config).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_dataset = H36MExpmapWindowDataset(
        cfg.train_npz_path,
        cfg.train_subjects,
        input_len,
        output_len,
        sample_rate=cfg.sample_rate,
    )
    train_batch_sampler = SequenceWindowBatchSampler(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle_sequences=True,
        shuffle_windows=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print("Train batches are sequence-pure: one batch contains windows from one synthetic sequence only.")
    print("Adaptive batch sampler:", "batch_size <=", cfg.batch_size, "num_batches =", len(train_batch_sampler))

    log_path = resolve_path(cfg.log_path)
    ensure_dir(os.path.dirname(log_path))
    acc_log = open(log_path, "w")
    acc_log.write("Seed : {}\n".format(cfg.seed))
    acc_log.write("Train NPZ : {}\n".format(cfg.train_npz_path))
    acc_log.write("Test NPZ : {}\n".format(cfg.test_npz_path))

    nb_iter = 0
    avg_loss = 0.0

    while nb_iter < cfg.total_iters:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step(model, past_expmap, future_expmap, optimizer, output_len)
            nb_iter += 1
            avg_loss += loss

            if nb_iter % cfg.print_every == 0:
                print("train iter {} loss={:.6f}".format(nb_iter, avg_loss / cfg.print_every))
                avg_loss = 0.0

            if nb_iter == cfg.total_iters:
                break

    metrics = evaluate_mpjpe(model, cfg.test_npz_path, cfg.test_subjects, config)
    print("Final MPJPE {}: {}".format(RESULT_KEYS, metrics))
    acc_log.write("final\n{}\n".format(" ".join(str(v) for v in metrics)))
    acc_log.flush()
    acc_log.close()

    print("Saved train/eval log to {}".format(log_path))
    return metrics


def main():
    distilled_path = train_distillation()
    train_and_evaluate(distilled_path)


if __name__ == "__main__":
    main()
