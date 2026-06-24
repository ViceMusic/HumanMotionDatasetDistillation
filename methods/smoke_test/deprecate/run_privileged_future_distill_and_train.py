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
# Privileged Future Guided Motion Distillation + train/eval
# Save as:
#   methods/smoke_test/run_privileged_future_distill_and_train.py
#
# Run:
#   cd ~/workspace/HumanMotionDatasetDistillation/methods/smoke_test
#   mamba activate HumanMotionDatasetDistillation
#   python run_privileged_future_distill_and_train.py
#
# Core idea:
#   Real window is split into:
#       O = observed sequence
#       Y = target sequence
#       P = privileged future sequence
#
#   Distillation learns one synthetic sequence per real sequence:
#       S = [S_obs, S_target, S_priv]
#
#   Gradient matching is only on:
#       O -> Y
#
#   Privileged future is NOT directly used as final prediction target.
#   It only provides motion trend / future velocity supervision:
#       L_priv_trend, L_priv_vel
#
#   Final saved training NPZ only contains:
#       [S_obs, S_target]
#
#   So downstream SiMLPe training remains the same prediction task:
#       50 frames -> 10 frames
# ============================================================


DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]
TEST_SUBJECTS = ["S5"]


@dataclass
class DistillConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"

    feature_dim: int = 99

    # SiMLPe default is 50 -> 10 during train.
    # Privileged window is target-after-future, not used as final train target.
    priv_len: int = 10

    batch_size: int = 8
    outer_steps: int = 8000

    num_backbones: int = 3
    backbone_reinit_interval: int = 20

    lr_synthetic: float = 1e-2

    # Fixed-style distillation weights.
    # No HDT harmonic loss here.
    lambda_grad: float = 0.1
    lambda_priv_trend: float = 0.05
    lambda_priv_vel: float = 0.05

    seed: int = 888
    print_interval: int = 10

    init_mode: str = "real"  # fixed: initialize each synthetic sequence from a real contiguous subseries

    save_name: str = "h36m_expmap_sequences_distilled_priv_future_g01_pt005_pv005_bs64_iter8000.npz"


@dataclass
class TrainConfig:
    train_npz_path: str = ""
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"

    log_path: str = "logs/train_priv_future_g01_pt005_pv005_bs64_iter8000.txt"

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


class SequencePrivilegedRealSampler:
    """
    Samples real contiguous windows from individual original sequences.

    Each original sequence has one corresponding learnable synthetic sequence.
    Real sample length is:
        input_len + target_len + priv_len
    """

    def __init__(self, data_path, distill_len, include_subjects=None, heldout_subjects=None):
        self.distill_len = distill_len
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

            if motion.shape[0] < distill_len:
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
            raise ValueError("No valid training sequences with length >= {} found in {}".format(distill_len, data_path))

    def sample(self, batch_size, keys=None):
        if keys is None:
            keys = random.sample(self.keys, k=min(batch_size, len(self.keys)))
            if len(keys) < batch_size:
                keys = keys + random.choices(self.keys, k=batch_size - len(keys))

        subs = []
        for key in keys:
            seq = self.by_key[str(key)]
            start = random.randint(0, seq.shape[0] - self.distill_len)
            subs.append(seq[start : start + self.distill_len])

        return torch.tensor(np.stack(subs), dtype=torch.float32), keys

    def make_initial_synthetic(self, init_mode="real"):
        if init_mode != "real":
            raise ValueError("This script uses fixed init_mode='real' for stable motion initialization.")

        init_motions = []
        for info in self.seq_infos:
            seq = self.by_key[info["key"]]
            start = random.randint(0, seq.shape[0] - self.distill_len)
            init_motions.append(seq[start : start + self.distill_len])
        return torch.tensor(np.stack(init_motions), dtype=torch.float32)

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


class SequenceTimeSyntheticMotionBank(nn.Module):
    """
    Time-domain synthetic bank indexed by original training sequence id.

    Row k corresponds to one real sequence.
    Synthetic sequence shape:
        [input_len + target_len + priv_len, 99]
    """

    def __init__(self, seq_infos, init_motions):
        super().__init__()
        self.seq_infos = list(seq_infos)
        self.keys = [info["key"] for info in self.seq_infos]
        self.key_to_idx = {key: idx for idx, key in enumerate(self.keys)}
        self.synthetic = nn.Parameter(init_motions.clone().float())

    @property
    def synthetic_len(self):
        return int(self.synthetic.shape[1])

    @property
    def feature_dim(self):
        return int(self.synthetic.shape[2])

    def forward(self, keys):
        ids = torch.tensor(
            [self.key_to_idx[str(key)] for key in keys],
            device=self.synthetic.device,
            dtype=torch.long,
        )
        return self.synthetic[ids]

    def get_all(self):
        return {
            "seq_infos": self.seq_infos,
            "synthetic_motions": self.synthetic.detach().cpu(),
        }


def split_oyp(series, input_len, target_len, priv_len):
    """
    series: [B, input_len + target_len + priv_len, C]
    return:
        obs, target, priv
    """
    obs = series[:, :input_len]
    target = series[:, input_len : input_len + target_len]
    priv = series[:, input_len + target_len : input_len + target_len + priv_len]
    return obs, target, priv


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
    Strict single-sequence gradient matching.

    Only O -> Y is used here.
    Privileged future P is not used as a prediction target.
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


def privileged_trend_velocity_loss(real_series, syn_series, input_len, target_len, priv_len, eps=1e-8):
    """
    Motion-specific privileged future guidance.

    It does NOT ask the model to predict privileged frames.
    It only makes synthetic future trend/velocity consistent with real future trend/velocity.

    Trend:
        mean(P) - last(O)

    Future velocity:
        mean velocity over [Y, P]
    """
    real_xyz = expmap_to_simlpe_xyz66(real_series)
    syn_xyz = expmap_to_simlpe_xyz66(syn_series)

    real_obs_last = real_xyz[:, input_len - 1]
    syn_obs_last = syn_xyz[:, input_len - 1]

    real_priv_mean = real_xyz[:, input_len + target_len : input_len + target_len + priv_len].mean(dim=1)
    syn_priv_mean = syn_xyz[:, input_len + target_len : input_len + target_len + priv_len].mean(dim=1)

    real_trend = real_priv_mean - real_obs_last
    syn_trend = syn_priv_mean - syn_obs_last

    real_trend = real_trend / (real_trend.norm(dim=-1, keepdim=True) + eps)
    syn_trend = syn_trend / (syn_trend.norm(dim=-1, keepdim=True) + eps)

    l_trend = F.mse_loss(syn_trend, real_trend.detach())

    real_future = real_xyz[:, input_len : input_len + target_len + priv_len]
    syn_future = syn_xyz[:, input_len : input_len + target_len + priv_len]

    real_vel = real_future[:, 1:] - real_future[:, :-1]
    syn_vel = syn_future[:, 1:] - syn_future[:, :-1]

    real_vel_mean = real_vel.mean(dim=1)
    syn_vel_mean = syn_vel.mean(dim=1)

    l_vel = F.mse_loss(syn_vel_mean, real_vel_mean.detach())

    return l_trend, l_vel


def build_random_backbones(config, num_backbones, device):
    backbones = []
    for _ in range(num_backbones):
        backbone = SiMLPeMotionBackbone(config).to(device)
        backbone.train()
        for param in backbone.parameters():
            param.requires_grad_(True)
        backbones.append(backbone)
    return backbones


def save_sequence_bank_npz(
    path,
    bank,
    step,
    logs,
    heldout_entries,
    train_save_len,
    feature_type="expmap",
    feature_dim=99,
):
    """
    Save only [S_obs, S_target] for downstream training.

    This is intentional:
      S_priv is used during distillation only.
      Final SiMLPe training must remain O -> Y.
    """
    ensure_dir(os.path.dirname(path))

    payload = bank.get_all()
    seq_infos = payload["seq_infos"]
    full_synthetic = payload["synthetic_motions"].numpy().astype(np.float32)
    synthetic_motions = full_synthetic[:, :train_save_len].astype(np.float32)

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
        raw_paths.append("distilled_priv_future://step_{}/{}".format(step, info["key"]))
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
        distill_method=np.array("privileged_future_guided_motion_distillation", dtype=object),
        train_save_len=np.array(train_save_len, dtype=np.int64),
        full_synthetic_len=np.array(full_synthetic.shape[1], dtype=np.int64),
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
    target_len = simlpe_config.motion.h36m_target_length_train
    priv_len = cfg.priv_len
    distill_len = input_len + target_len + priv_len
    train_save_len = input_len + target_len

    sampler = SequencePrivilegedRealSampler(
        cfg.data_path,
        distill_len=distill_len,
        include_subjects=DISTILL_SUBJECTS,
        heldout_subjects=HELDOUT_SUBJECTS,
    )

    init_motions = sampler.make_initial_synthetic(init_mode=cfg.init_mode)
    bank = SequenceTimeSyntheticMotionBank(sampler.seq_infos, init_motions).to(device)

    backbones = build_random_backbones(simlpe_config, cfg.num_backbones, device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=cfg.lr_synthetic)

    output_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(root_dir(), cfg.output_dir)
    save_path = os.path.join(output_dir, cfg.save_name)

    logs = []

    print("Device:", device)
    print("Method: Privileged Future Guided Motion Distillation")
    print("Num training sequences:", len(sampler.keys))
    print("Input length:", input_len, "Target length:", target_len, "Priv length:", priv_len)
    print("Internal synthetic length:", distill_len, "Saved train length:", train_save_len)
    print("lambda_grad:", cfg.lambda_grad)
    print("lambda_priv_trend:", cfg.lambda_priv_trend)
    print("lambda_priv_vel:", cfg.lambda_priv_vel)
    print("Save path:", save_path)

    for step in range(1, cfg.outer_steps + 1):
        if cfg.backbone_reinit_interval > 0 and step > 1 and (step - 1) % cfg.backbone_reinit_interval == 0:
            backbones = build_random_backbones(simlpe_config, cfg.num_backbones, device)

        real_batch, keys = sampler.sample(cfg.batch_size)
        real_batch = real_batch.to(device)
        syn_batch = bank(keys)

        per_seq_losses = []
        per_seq_grads = []
        per_seq_trends = []
        per_seq_vels = []

        # Strict sequence-level loss:
        # real_i only compares with syn_i.
        # P is only trend/velocity guidance, not prediction target.
        for i in range(real_batch.shape[0]):
            real_series = real_batch[i : i + 1]
            syn_series = syn_batch[i : i + 1]

            real_obs, real_target, _real_priv = split_oyp(real_series, input_len, target_len, priv_len)
            syn_obs, syn_target, _syn_priv = split_oyp(syn_series, input_len, target_len, priv_len)

            grad_losses = [
                compute_gradient_matching_loss_single(
                    backbone,
                    real_obs,
                    real_target,
                    syn_obs,
                    syn_target,
                    target_len,
                )
                for backbone in backbones
            ]
            l_grad = torch.stack(grad_losses).mean()

            l_priv_trend, l_priv_vel = privileged_trend_velocity_loss(
                real_series,
                syn_series,
                input_len=input_len,
                target_len=target_len,
                priv_len=priv_len,
            )

            loss_i = (
                cfg.lambda_grad * l_grad
                + cfg.lambda_priv_trend * l_priv_trend
                + cfg.lambda_priv_vel * l_priv_vel
            )

            per_seq_losses.append(loss_i)
            per_seq_grads.append(cfg.lambda_grad * l_grad.detach())
            per_seq_trends.append(cfg.lambda_priv_trend * l_priv_trend.detach())
            per_seq_vels.append(cfg.lambda_priv_vel * l_priv_vel.detach())

        total_loss = torch.stack(per_seq_losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            syn_time = bank.synthetic
            syn_mean = float(syn_time.mean().detach().cpu())
            syn_std = float(syn_time.std().detach().cpu())
            syn_min = float(syn_time.min().detach().cpu())
            syn_max = float(syn_time.max().detach().cpu())

        row = {
            "step": step,
            "L_grad": float(torch.stack(per_seq_grads).mean().detach().cpu()),
            "L_priv_trend": float(torch.stack(per_seq_trends).mean().detach().cpu()),
            "L_priv_vel": float(torch.stack(per_seq_vels).mean().detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
            "syn_mean": syn_mean,
            "syn_std": syn_std,
            "syn_min": syn_min,
            "syn_max": syn_max,
            "example_key": keys[0],
        }
        logs.append(row)

        if step % cfg.print_interval == 0:
            print(
                "distill step {step} "
                "L_grad={L_grad:.6f} "
                "L_priv_trend={L_priv_trend:.6f} "
                "L_priv_vel={L_priv_vel:.6f} "
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
        train_save_len=train_save_len,
        feature_dim=cfg.feature_dim,
    )

    print("Saved privileged-future synthetic bank to {}".format(save_path))
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

    For this script, synthetic train sequences are saved as exactly input_len+target_len,
    so each synthetic sequence contributes exactly one window.
    Heldout S5 is carried in the NPZ but not used for training.
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
    print("Synthetic train NPZ saves only [S_obs, S_target]. Privileged frames are not used for final train.")
    print("Adaptive batch sampler:", "batch_size <=", cfg.batch_size, "num_batches =", len(train_batch_sampler))

    log_path = resolve_path(cfg.log_path)
    ensure_dir(os.path.dirname(log_path))
    acc_log = open(log_path, "w")
    acc_log.write("Seed : {}\n".format(cfg.seed))
    acc_log.write("Method : privileged_future_guided_motion_distillation\n")
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
