#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GM seed sweep for Human3.6M / siMLPe.

Experiments in this script:
  1) random100_gm      x seeds [888, 999, 1234, 2026, 3407]
  2) middle100_gm      x seeds [888, 999, 1234, 2026, 3407]
  3) uniform_piece5x20_gm x seeds [888, 999, 1234, 2026, 3407]

Each run writes an individual log under LOG_ROOT. The final summary reports
per-method mean ± std over the five seeds and is written to SUMMARY_LOG.
Experiments are launched in parallel processes.

Save as:
  methods/smoke_test/run_gm_seed_sweep_parallel.py
Run:
  python methods/smoke_test/run_gm_seed_sweep_parallel.py
"""

import os
import sys
import json
import time
import random
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# Global experiment setup
# ============================================================

ALL_SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]
TEST_SUBJECTS = ["S5"]

RESULT_KEYS = ["#2", "#4", "#8", "#10", "#14", "#18", "#22", "#25"]
JOINT_USED_XYZ = np.array(
    [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
).astype(np.int64)
JOINT_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
JOINT_EQUAL = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)


@dataclass
class GlobalConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets/gm_seed_sweep_parallel"
    log_root: str = "logs/gm_seed_sweep_parallel"
    summary_log: str = "logs/gm_seed_sweep_parallel/summary_mean_std.txt"

    synthetic_len: int = 100
    feature_dim: int = 99
    seed: int = 888
    seeds = (888, 999, 1234, 2026, 3407)

    # 15 runs total. A6000 should usually hold this, but reduce if CUDA OOM.
    max_workers: int = 15


@dataclass
class GMConfig:
    batch_size: int = 16
    outer_steps: int = 8000
    num_backbones: int = 1
    lr_synthetic: float = 1e-2

    # This is the current "our GM" setting: raw sequence + GM + pred/vel regularization.
    lambda_grad: float = 0.1
    lambda_pred: float = 150.0
    lambda_vel: float = 150.0

    window_mode: str = "random"
    num_windows_per_sequence: int = 1
    print_interval: int = 10


@dataclass
class TrainConfig:
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    train_subjects = DISTILL_SUBJECTS
    test_subjects = TEST_SUBJECTS

    sample_rate: int = 1
    batch_size: int = 64
    num_workers: int = 1
    total_iters: int = 8000
    print_every: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4


BASE_GM_EXPERIMENTS = [
    {
        "base_name": "random100_gm",
        "segment_method": "random100",
        "desc": "Seed sweep GM refinement: initialize each synthetic sequence with one random contiguous 100-frame crop, optimize it with siMLPe GM + pred/vel regularization, then train/evaluate siMLPe.",
    },
    {
        "base_name": "middle100_gm",
        "segment_method": "middle100",
        "desc": "Seed sweep GM refinement: initialize each synthetic sequence with the middle contiguous 100 frames, optimize it with siMLPe GM + pred/vel regularization, then train/evaluate siMLPe.",
    },
    {
        "base_name": "uniform_piece5x20_gm",
        "segment_method": "uniform_piece5x20",
        "desc": "Seed sweep GM refinement: initialize each synthetic sequence with 20 uniformly spaced chronological pieces of length 5, optimize it with siMLPe GM + pred/vel regularization, then train/evaluate siMLPe.",
    },
]


def build_seed_sweep_experiments():
    global_cfg = GlobalConfig()
    exps = []
    for base in BASE_GM_EXPERIMENTS:
        for seed in global_cfg.seeds:
            exps.append({
                "name": "{}_seed{}".format(base["base_name"], seed),
                "base_name": base["base_name"],
                "segment_method": base["segment_method"],
                "use_gm": True,
                "seed": int(seed),
                "desc": base["desc"],
            })
    return exps



# ============================================================
# Utilities
# ============================================================

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Segment construction
# ============================================================

def sample_segment(seq, out_len, method):
    n = seq.shape[0]
    if n < out_len:
        raise ValueError("Sequence length {} < out_len {}".format(n, out_len))

    if method == "first100":
        start = 0
        return seq[start:start + out_len].copy(), {"mode": method, "start": start}

    if method == "middle100":
        start = (n - out_len) // 2
        return seq[start:start + out_len].copy(), {"mode": method, "start": start}

    if method == "last100":
        start = n - out_len
        return seq[start:start + out_len].copy(), {"mode": method, "start": start}

    if method == "random100":
        start = random.randint(0, n - out_len)
        return seq[start:start + out_len].copy(), {"mode": method, "start": start}

    if method == "uniform_piece5x20":
        return sample_uniform_pieces(seq, out_len=out_len, piece_len=5, num_pieces=20, method=method)

    if method == "uniform_piece10x10":
        return sample_uniform_pieces(seq, out_len=out_len, piece_len=10, num_pieces=10, method=method)

    raise ValueError("Unknown segment method: {}".format(method))


def sample_uniform_pieces(seq, out_len, piece_len, num_pieces, method):
    if piece_len * num_pieces != out_len:
        raise ValueError("piece_len*num_pieces must equal out_len")
    n = seq.shape[0]
    if n < piece_len:
        raise ValueError("Sequence length {} < piece_len {}".format(n, piece_len))

    if num_pieces == 1:
        starts = [0]
    else:
        starts = np.linspace(0, n - piece_len, num_pieces, dtype=np.int64).tolist()
        starts = [int(s) for s in starts]
    pieces = [seq[s:s + piece_len] for s in starts]
    out = np.concatenate(pieces, axis=0).astype(np.float32)
    return out, {"mode": method, "piece_len": piece_len, "num_pieces": num_pieces, "starts_head": starts[:10]}


class OriginalSequencePool:
    def __init__(self, data_path, include_subjects=None, heldout_subjects=None, min_len=100):
        self.data = np.load(data_path, allow_pickle=True)
        include_subjects = set(include_subjects or [])
        heldout_subjects = set(heldout_subjects or [])
        self.seq_infos, self.by_key, self.heldout_entries = [], {}, []

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
            if motion.ndim != 2 or motion.shape[1] != 99 or motion.shape[0] < min_len:
                continue
            key, subject_name, action_name, trial_name, raw_path_text = make_seq_key(
                idx, subject_name, action, trial, raw_path
            )
            self.by_key[key] = motion
            self.seq_infos.append({
                "key": key,
                "subject": subject_name,
                "action": action_name,
                "trial": trial_name,
                "raw_path": raw_path_text,
                "original_length": motion.shape[0],
            })
        self.keys = [info["key"] for info in self.seq_infos]
        if not self.keys:
            raise ValueError("No valid training sequences found in {}".format(data_path))

    def get_heldout_entries(self):
        entries = []
        for idx in self.heldout_entries:
            motion = np.asarray(self.data["motions"][idx], dtype=np.float32)
            entries.append({
                "subject": self.data["subjects"][idx],
                "action": self.data["actions"][idx],
                "trial": self.data["trials"][idx],
                "length": motion.shape[0],
                "raw_path": self.data["raw_paths"][idx],
                "motion": motion,
            })
        return entries

    def build_initial_segments(self, method, out_len):
        segments, metas = [], []
        for info in self.seq_infos:
            key = info["key"]
            seg, meta = sample_segment(self.by_key[key], out_len, method)
            segments.append(seg)
            metas.append({"key": key, "method": method, "original_length": info["original_length"], "meta": meta})
        return torch.tensor(np.stack(segments), dtype=torch.float32), metas


# ============================================================
# Synthetic bank
# ============================================================

class SequenceFrequencySyntheticMotionBank(nn.Module):
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
                raise ValueError("init_motion shape {} does not match expected {}".format(tuple(init_motion.shape), expected_shape))

        init_freq = torch.fft.rfft(init_motion, dim=1)
        self.freq_real = nn.Parameter(init_freq.real)
        self.freq_imag = nn.Parameter(init_freq.imag)

    def get_freq(self):
        return torch.complex(self.freq_real, self.freq_imag)

    def get_time(self):
        return torch.fft.irfft(self.get_freq(), n=self.synthetic_len, dim=1)

    def forward(self, keys):
        all_motion = self.get_time()
        ids = torch.tensor([self.key_to_idx[str(key)] for key in keys], device=all_motion.device, dtype=torch.long)
        return all_motion[ids]

    @torch.no_grad()
    def project_valid_rfft(self):
        self.freq_imag[:, 0, :].zero_()
        if self.synthetic_len % 2 == 0:
            self.freq_imag[:, -1, :].zero_()

    def get_all(self):
        return {"seq_infos": self.seq_infos, "synthetic_motions": self.get_time().detach().cpu()}


# ============================================================
# Save NPZ
# ============================================================

def save_npz_from_motions(path, seq_infos, motions, heldout_entries, meta_array=None, feature_type="expmap", feature_dim=99):
    ensure_dir(os.path.dirname(path))
    subjects, actions, trials, lengths, raw_paths = [], [], [], [], []

    for info, motion in zip(seq_infos, motions):
        subjects.append(info["subject"])
        actions.append(info["action"])
        trials.append(info["trial"])
        lengths.append(motion.shape[0])
        raw_paths.append(info.get("save_raw_path", "segment://{}".format(info["key"])))

    motion_list = [np.asarray(m, dtype=np.float32) for m in motions]

    for entry in heldout_entries:
        subjects.append(entry["subject"])
        actions.append(entry["action"])
        trials.append(entry["trial"])
        lengths.append(entry["length"])
        raw_paths.append(entry["raw_path"])
        motion_list.append(entry["motion"])

    motion_array = np.empty(len(motion_list), dtype=object)
    for idx, motion in enumerate(motion_list):
        motion_array[idx] = motion

    kwargs = {}
    if meta_array is not None:
        kwargs["segment_meta"] = np.array(meta_array, dtype=object)

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
        **kwargs,
    )
    return path


def save_baseline_npz(path, pool, method, synthetic_len, feature_dim):
    init_motion, metas = pool.build_initial_segments(method, synthetic_len)
    seq_infos = []
    for info in pool.seq_infos:
        item = dict(info)
        item["save_raw_path"] = "segment://{}/{}".format(method, info["key"])
        seq_infos.append(item)
    return save_npz_from_motions(
        path,
        seq_infos,
        init_motion.numpy().astype(np.float32),
        heldout_entries=pool.get_heldout_entries(),
        meta_array=metas,
        feature_dim=feature_dim,
    )


def save_bank_npz(path, bank, step, logs, heldout_entries, init_method, feature_type="expmap", feature_dim=99):
    payload = bank.get_all()
    seq_infos = []
    for info in payload["seq_infos"]:
        item = dict(info)
        item["save_raw_path"] = "gm://{}_step_{}/{}".format(init_method, step, info["key"])
        seq_infos.append(item)
    motions = payload["synthetic_motions"].numpy().astype(np.float32)
    meta = [{"gm_step": step, "init_method": init_method, "key": info["key"]} for info in seq_infos]
    path = save_npz_from_motions(path, seq_infos, motions, heldout_entries, meta_array=meta, feature_type=feature_type, feature_dim=feature_dim)

    # Store logs in a companion npy to avoid huge object payload in npz.
    np.save(path + ".distill_logs.npy", np.array(logs, dtype=object), allow_pickle=True)
    return path


# ============================================================
# Distillation
# ============================================================

def make_windows_from_subseries(series, input_len, output_len, num_windows=None, window_mode="random"):
    batch_size, synthetic_len, _ = series.shape
    total_len = input_len + output_len
    if synthetic_len < total_len:
        raise ValueError("synthetic_len={} is shorter than input_len+output_len={}".format(synthetic_len, total_len))
    windows_per_series = 1 if num_windows is None else num_windows
    max_start = synthetic_len - total_len

    past_windows, future_windows = [], []
    for batch_idx in range(batch_size):
        for _ in range(windows_per_series):
            if window_mode == "first":
                start = 0
            elif window_mode == "random":
                start = torch.randint(0, max_start + 1, (1,), device=series.device).item()
            else:
                raise ValueError("Unknown window_mode: {}".format(window_mode))
            window = series[batch_idx, start:start + total_len]
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


def train_distillation_from_segments(pool, method, save_path, gcfg, global_cfg, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simlpe_config = build_simlpe_config()
    input_len = simlpe_config.motion.h36m_input_length
    output_len = simlpe_config.motion.h36m_target_length_train

    init_motion, _metas = pool.build_initial_segments(method, global_cfg.synthetic_len)
    bank = SequenceFrequencySyntheticMotionBank(
        pool.seq_infos,
        synthetic_len=global_cfg.synthetic_len,
        feature_dim=global_cfg.feature_dim,
        init_motion=init_motion,
    ).to(device)
    bank.project_valid_rfft()

    backbones = build_random_backbones(simlpe_config, gcfg.num_backbones, device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=gcfg.lr_synthetic)
    logs = []

    print("[GM] init method:", method)
    print("[GM] save path:", save_path)
    print("[GM] device:", device)
    print("[GM] input_len/output_len:", input_len, output_len)
    print("[GM] lambda_grad/pred/vel:", gcfg.lambda_grad, gcfg.lambda_pred, gcfg.lambda_vel)

    for step in range(1, gcfg.outer_steps + 1):
        keys = random.sample(pool.keys, k=min(gcfg.batch_size, len(pool.keys)))
        if len(keys) < gcfg.batch_size:
            keys = keys + random.choices(pool.keys, k=gcfg.batch_size - len(keys))

        real_batch = []
        for key in keys:
            seg, _ = sample_segment(pool.by_key[str(key)], global_cfg.synthetic_len, method)
            real_batch.append(seg)
        real_sub_batch = torch.tensor(np.stack(real_batch), dtype=torch.float32, device=device)
        syn_sub_batch = bank(keys)

        per_seq_losses, per_seq_grads, per_seq_preds, per_seq_vels = [], [], [], []

        for i in range(real_sub_batch.shape[0]):
            real_sub = real_sub_batch[i:i + 1]
            syn_sub = syn_sub_batch[i:i + 1]

            real_past, real_future = make_windows_from_subseries(
                real_sub, input_len, output_len,
                num_windows=gcfg.num_windows_per_sequence,
                window_mode=gcfg.window_mode,
            )
            syn_past, syn_future = make_windows_from_subseries(
                syn_sub, input_len, output_len,
                num_windows=gcfg.num_windows_per_sequence,
                window_mode=gcfg.window_mode,
            )

            grad_losses = [
                compute_gradient_matching_loss_single(
                    backbone, real_past, real_future, syn_past, syn_future, output_len
                )
                for backbone in backbones
            ]
            l_grad = torch.stack(grad_losses).mean()

            syn_pred = backbones[0](syn_past, output_len=output_len)
            l_pred_syn = prediction_loss(syn_pred, syn_future)
            l_vel_syn = velocity_loss(syn_pred, syn_future)

            loss_i = (
                gcfg.lambda_grad * l_grad
                + gcfg.lambda_pred * l_pred_syn
                + gcfg.lambda_vel * l_vel_syn
            )
            per_seq_losses.append(loss_i)
            per_seq_grads.append(gcfg.lambda_grad * l_grad.detach())
            per_seq_preds.append(gcfg.lambda_pred * l_pred_syn.detach())
            per_seq_vels.append(gcfg.lambda_vel * l_vel_syn.detach())

        total_loss = torch.stack(per_seq_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        bank.project_valid_rfft()

        with torch.no_grad():
            syn_time = bank.get_time()
            row = {
                "step": step,
                "L_grad": float(torch.stack(per_seq_grads).mean().detach().cpu()),
                "L_pred_syn": float(torch.stack(per_seq_preds).mean().detach().cpu()),
                "L_vel_syn": float(torch.stack(per_seq_vels).mean().detach().cpu()),
                "L_total": float(total_loss.detach().cpu()),
                "syn_mean": float(syn_time.mean().detach().cpu()),
                "syn_std": float(syn_time.std().detach().cpu()),
                "syn_min": float(syn_time.min().detach().cpu()),
                "syn_max": float(syn_time.max().detach().cpu()),
                "example_key": keys[0],
            }
        logs.append(row)

        if step % gcfg.print_interval == 0:
            print(
                "distill step {step} L_grad={L_grad:.6f} L_pred_syn={L_pred_syn:.6f} "
                "L_vel_syn={L_vel_syn:.6f} L_total={L_total:.6f} "
                "syn_mean={syn_mean:.6f} syn_std={syn_std:.6f} example_key={example_key}".format(**row),
                flush=True,
            )

    save_bank_npz(
        save_path,
        bank,
        gcfg.outer_steps,
        logs,
        heldout_entries=pool.get_heldout_entries(),
        init_method=method,
        feature_dim=global_cfg.feature_dim,
    )
    print("[GM] Saved distilled dataset to", save_path)
    return save_path


# ============================================================
# Train/eval
# ============================================================

class H36MExpmapWindowDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=1):
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.seqs, self.data_idx = [], []
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
        window = self.seqs[seq_idx][start:start + self.total_len]
        return window[:self.input_len], window[self.input_len:]


class SequenceWindowBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle_sequences=True, shuffle_windows=True):
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
                batch = indices[start:start + self.batch_size]
                if batch:
                    yield batch

    def __len__(self):
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.seq_to_indices.values())


class H36MExpmapEvalDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=1):
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.seqs, self.data_idx = [], []
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
        window = self.seqs[seq_idx][start:start + self.total_len]
        motion_xyz32 = expmap_to_xyz32(window.unsqueeze(0)).squeeze(0) / 1000.0
        return window[:self.input_len], motion_xyz32[self.input_len:]


def evaluate_mpjpe(model, npz_path, subjects, config):
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_eval
    dataset = H36MExpmapEvalDataset(npz_path, subjects, input_len, output_len)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

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
    ret = {"#{}".format(idx + 1): mpjpe_by_frame[idx] for idx in range(output_len)}
    return [round(float(ret[key]), 1) for key in RESULT_KEYS]


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


def train_and_evaluate(train_npz_path, method_name, train_cfg, seed):
    set_seed(seed)
    config = build_simlpe_config()
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_train

    model = SiMLPeMotionBackbone(config).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    train_dataset = H36MExpmapWindowDataset(
        train_npz_path,
        train_cfg.train_subjects,
        input_len,
        output_len,
        sample_rate=train_cfg.sample_rate,
    )
    train_batch_sampler = SequenceWindowBatchSampler(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle_sequences=True,
        shuffle_windows=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    print("[TRAIN] method:", method_name)
    print("[TRAIN] train_npz:", train_npz_path)
    print("[TRAIN] num_windows:", len(train_dataset), "num_batches:", len(train_batch_sampler))
    print("[TRAIN] batch_size:", train_cfg.batch_size, "total_iters:", train_cfg.total_iters)

    nb_iter, avg_loss = 0, 0.0
    while nb_iter < train_cfg.total_iters:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step(model, past_expmap, future_expmap, optimizer, output_len)
            nb_iter += 1
            avg_loss += loss
            if nb_iter % train_cfg.print_every == 0:
                print("[{}] train iter {} loss={:.6f}".format(method_name, nb_iter, avg_loss / train_cfg.print_every), flush=True)
                avg_loss = 0.0
            if nb_iter == train_cfg.total_iters:
                break

    metrics = evaluate_mpjpe(model, train_cfg.test_npz_path, train_cfg.test_subjects, config)
    print("[{}] Final MPJPE {}: {}".format(method_name, RESULT_KEYS, metrics))
    return metrics


# ============================================================
# Per-experiment process
# ============================================================

def run_one_experiment(exp):
    global_cfg = GlobalConfig()
    gcfg = GMConfig()
    train_cfg = TrainConfig()

    seed = int(exp.get("seed", global_cfg.seed))

    output_dir = resolve_path(global_cfg.output_dir)
    log_root = resolve_path(global_cfg.log_root)
    ensure_dir(output_dir)
    ensure_dir(log_root)

    exp_log_path = os.path.join(log_root, "{}.log".format(exp["name"]))
    exp_npz_path = os.path.join(output_dir, "h36m_{}_{}.npz".format(exp["name"], global_cfg.synthetic_len))

    with open(exp_log_path, "w", buffering=1) as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            print("Experiment:", exp["name"])
            print("Description:", exp["desc"])
            print("Base experiment:", exp.get("base_name", exp["name"]))
            print("Segment method:", exp["segment_method"])
            print("Use GM:", exp["use_gm"])
            print("Seed:", seed)
            print("Output NPZ:", exp_npz_path)
            print("Individual log:", exp_log_path)
            print("RESULT_KEYS:", RESULT_KEYS)
            print("=" * 80, flush=True)

            started = time.time()
            try:
                set_seed(seed)
                pool = OriginalSequencePool(
                    global_cfg.data_path,
                    include_subjects=DISTILL_SUBJECTS,
                    heldout_subjects=HELDOUT_SUBJECTS,
                    min_len=global_cfg.synthetic_len,
                )
                print("Num training sequences:", len(pool.keys))
                print("Synthetic length:", global_cfg.synthetic_len)

                if exp["use_gm"]:
                    train_npz_path = train_distillation_from_segments(
                        pool=pool,
                        method=exp["segment_method"],
                        save_path=exp_npz_path,
                        gcfg=gcfg,
                        global_cfg=global_cfg,
                        seed=seed,
                    )
                else:
                    train_npz_path = save_baseline_npz(
                        exp_npz_path,
                        pool=pool,
                        method=exp["segment_method"],
                        synthetic_len=global_cfg.synthetic_len,
                        feature_dim=global_cfg.feature_dim,
                    )
                    print("[BASELINE] Saved baseline dataset to", train_npz_path)

                metrics = train_and_evaluate(train_npz_path, exp["name"], train_cfg, seed)
                elapsed = time.time() - started
                print("Elapsed seconds:", round(elapsed, 2))

                result = {
                    "name": exp["name"],
                    "base_name": exp.get("base_name", exp["name"]),
                    "seed": seed,
                    "segment_method": exp["segment_method"],
                    "use_gm": exp["use_gm"],
                    "metrics": metrics,
                    "log_path": exp_log_path,
                    "npz_path": train_npz_path,
                    "elapsed_sec": elapsed,
                    "status": "ok",
                }
                print("JSON_RESULT:", json.dumps(result, ensure_ascii=False))
                return result
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "name": exp["name"],
                    "base_name": exp.get("base_name", exp["name"]),
                    "seed": seed,
                    "segment_method": exp["segment_method"],
                    "use_gm": exp["use_gm"],
                    "metrics": None,
                    "log_path": exp_log_path,
                    "npz_path": exp_npz_path,
                    "status": "failed",
                    "error": repr(exc),
                }
                print("JSON_RESULT:", json.dumps(result, ensure_ascii=False))
                return result


# ============================================================
# Main parallel runner
# ============================================================

def main():
    # Spawn is safer with CUDA than fork.
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    global_cfg = GlobalConfig()
    ensure_dir(resolve_path(global_cfg.log_root))
    ensure_dir(os.path.dirname(resolve_path(global_cfg.summary_log)))

    summary_path = resolve_path(global_cfg.summary_log)
    experiments = build_seed_sweep_experiments()
    print("Running {} experiments with max_workers={}".format(len(experiments), global_cfg.max_workers))
    print("Seeds:", list(global_cfg.seeds))
    print("Summary log:", summary_path)

    results = []
    with ProcessPoolExecutor(max_workers=global_cfg.max_workers) as executor:
        futures = {executor.submit(run_one_experiment, exp): exp for exp in experiments}
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "name": exp["name"],
                    "base_name": exp.get("base_name", exp["name"]),
                    "seed": exp.get("seed"),
                    "segment_method": exp["segment_method"],
                    "use_gm": exp["use_gm"],
                    "metrics": None,
                    "status": "failed",
                    "error": repr(exc),
                }
            results.append(result)
            print("Finished:", result["name"], "status=", result.get("status"), "metrics=", result.get("metrics"), flush=True)

            # Continuously update summary in case the run is interrupted.
            write_summary(summary_path, results)

    # Final ordered summary.
    order = {exp["name"]: i for i, exp in enumerate(experiments)}
    results = sorted(results, key=lambda r: order.get(r["name"], 999))
    write_summary(summary_path, results)
    print("Done. Summary written to", summary_path)


def write_summary(summary_path, results):
    ensure_dir(os.path.dirname(summary_path))
    ordered_bases = [base["base_name"] for base in BASE_GM_EXPERIMENTS]

    grouped = {base: [] for base in ordered_bases}
    for r in results:
        base = r.get("base_name", r.get("name"))
        grouped.setdefault(base, []).append(r)

    with open(summary_path, "w") as f:
        f.write("GM seed sweep summary for Human3.6M / siMLPe.\n")
        f.write("This run evaluates three GM-refined initialization methods over five seeds.\n")
        f.write("Each individual log begins with a short description of the experiment and stores full distillation/train output.\n")
        f.write("GM setting: siMLPe gradient matching + pred/vel regularization, raw sequence, one synthetic sequence per real sequence.\n")
        f.write("RESULT_KEYS: {}\n".format(RESULT_KEYS))
        f.write("Seeds: {}\n".format(list(GlobalConfig().seeds)))
        f.write("\n")

        f.write("Per-run results:\n")
        for r in sorted(results, key=lambda x: (x.get("base_name", x.get("name")), int(x.get("seed") or 0))):
            f.write("{}\n".format(r.get("name")))
            f.write("  status: {}\n".format(r.get("status")))
            f.write("  base_name: {}\n".format(r.get("base_name")))
            f.write("  seed: {}\n".format(r.get("seed")))
            f.write("  segment_method: {}\n".format(r.get("segment_method")))
            f.write("  metrics: {}\n".format(r.get("metrics")))
            f.write("  log_path: {}\n".format(r.get("log_path")))
            f.write("  npz_path: {}\n".format(r.get("npz_path")))
            if r.get("error"):
                f.write("  error: {}\n".format(r.get("error")))
            f.write("\n")

        f.write("Mean ± std by method:\n")
        for base in ordered_bases:
            ok = [r for r in grouped.get(base, []) if r.get("status") == "ok" and r.get("metrics") is not None]
            f.write("{}\n".format(base))
            f.write("  num_ok: {} / {}\n".format(len(ok), len(GlobalConfig().seeds)))
            if ok:
                arr = np.array([r["metrics"] for r in ok], dtype=np.float64)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0, ddof=0)
                f.write("  mean: [{}]\n".format(", ".join("{:.2f}".format(x) for x in mean)))
                f.write("  std:  [{}]\n".format(", ".join("{:.2f}".format(x) for x in std)))
                f.write("  mean±std: [{}]\n".format(", ".join("{:.2f}±{:.2f}".format(m, sd) for m, sd in zip(mean, std))))
            f.write("\n")


if __name__ == "__main__":
    main()
