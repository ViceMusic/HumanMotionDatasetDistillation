#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DLinear seed sweep for Human3.6M motion forecasting.

Experiments per seed:
  1) full                         : train DLinear on the original full training set, no distillation.
  2) distilled_dlinear_gm          : random contiguous init + DLinear GM + pred/vel regularization.
  3) distilled_dlinear_gm_pure     : random contiguous init + DLinear GM only; lambda_pred=lambda_vel=0.
  4) random_complete               : one random contiguous 100-frame crop per training sequence, no distillation.
  5) random_20pieces               : 20 random pieces concatenated to 100 frames per training sequence, no distillation.
  6) random_20pieces_gm            : random_20pieces init + DLinear GM + pred/vel regularization.

Each run writes an individual log. The final summary reports mean ± std over seeds.

Save as:
  methods/smoke_test/run_dlinear_seed_sweep_full_random_gm.py
Run:
  python methods/smoke_test/run_dlinear_seed_sweep_full_random_gm.py
"""

import os
import json
import time
import random
import traceback
from dataclasses import dataclass
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from backbones.simlpe import (
    build_simlpe_config,
    expmap_to_xyz32,
)


# ============================================================
# Experiment setup
# ============================================================

DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]
TEST_SUBJECTS = ["S5"]
RESULT_KEYS = ["#2", "#4", "#8", "#10", "#14", "#18", "#22", "#25"]


@dataclass
class GlobalConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets/dlinear_seed_sweep_full_random_gm_protocol_fixed"
    log_root: str = "logs/dlinear_seed_sweep_full_random_gm_protocol_fixed"
    summary_log: str = "logs/dlinear_seed_sweep_full_random_gm_protocol_fixed/summary_mean_std.txt"

    synthetic_len: int = 100
    feature_dim: int = 99
    seeds = (888, 999, 1234, 2026, 3407)

    # DLinear is light. Reduce this if your CPU/DataLoader becomes the bottleneck.
    max_workers: int = 30


@dataclass
class DistillConfig:
    batch_size: int = 16
    outer_steps: int = 8000
    num_backbones: int = 1
    lr_synthetic: float = 1e-2

    lambda_harm: float = 0.0
    lambda_grad: float = 0.1
    lambda_pred: float = 150.0
    lambda_vel: float = 150.0

    use_hdt_filter: bool = False
    top_k: int = 16
    window_mode: str = "random"
    print_interval: int = 10


@dataclass
class TrainConfig:
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    train_subjects = DISTILL_SUBJECTS
    test_subjects = TEST_SUBJECTS

    sample_rate: int = 1
    batch_size: int = 64
    num_workers: int = 4
    total_iters: int = 8000
    print_every: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4


BASE_EXPERIMENTS = [
    {
        "base_name": "full",
        "mode": "full",
        "init_method": None,
        "use_gm": False,
        "lambda_pred": None,
        "lambda_vel": None,
        "desc": "Full-data baseline: train DLinear on the original full Human3.6M training set; no distillation and no random compression.",
    },
    {
        "base_name": "distilled_dlinear_gm",
        "mode": "gm",
        "init_method": "random_complete",
        "use_gm": True,
        "lambda_pred": 150.0,
        "lambda_vel": 150.0,
        "desc": "DLinear GM: initialize each synthetic sequence with one random contiguous 100-frame crop, then run DLinear GM + pred/vel regularization before train/eval.",
    },
    {
        "base_name": "distilled_dlinear_gm_pure",
        "mode": "gm",
        "init_method": "random_complete",
        "use_gm": True,
        "lambda_pred": 0.0,
        "lambda_vel": 0.0,
        "desc": "Pure DLinear GM: initialize with random contiguous 100-frame crops, then run DLinear GM only; lambda_pred=lambda_vel=0 during distillation.",
    },
    {
        "base_name": "random_complete",
        "mode": "baseline",
        "init_method": "random_complete",
        "use_gm": False,
        "lambda_pred": None,
        "lambda_vel": None,
        "desc": "Random baseline: one random contiguous 100-frame crop per training sequence; no distillation.",
    },
    {
        "base_name": "random_20pieces",
        "mode": "baseline",
        "init_method": "random_20pieces",
        "use_gm": False,
        "lambda_pred": None,
        "lambda_vel": None,
        "desc": "Random baseline: 20 random pieces concatenated into 100 frames per training sequence; no distillation.",
    },
    {
        "base_name": "random_20pieces_gm",
        "mode": "gm",
        "init_method": "random_20pieces",
        "use_gm": True,
        "lambda_pred": 150.0,
        "lambda_vel": 150.0,
        "desc": "DLinear GM from fragmented init: initialize each synthetic sequence with random_20pieces, then run DLinear GM + pred/vel regularization before train/eval.",
    },
]


def build_experiments():
    global_cfg = GlobalConfig()
    exps = []
    for base in BASE_EXPERIMENTS:
        for seed in global_cfg.seeds:
            exp = dict(base)
            exp["seed"] = int(seed)
            exp["name"] = f"{base['base_name']}_seed{seed}"
            exps.append(exp)
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
# DLinear backbone
# ============================================================


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        pad_front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        pad_end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([pad_front, x, pad_end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinearMotionBackbone(nn.Module):
    def __init__(self, input_len, output_len, feature_dim=99, moving_avg=25, individual=False):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.feature_dim = feature_dim
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(input_len, output_len) for _ in range(feature_dim)])
            self.linear_trend = nn.ModuleList([nn.Linear(input_len, output_len) for _ in range(feature_dim)])
            for i in range(feature_dim):
                self.linear_seasonal[i].weight.data.fill_(1.0 / input_len)
                self.linear_trend[i].weight.data.fill_(1.0 / input_len)
        else:
            self.linear_seasonal = nn.Linear(input_len, output_len)
            self.linear_trend = nn.Linear(input_len, output_len)
            self.linear_seasonal.weight.data.fill_(1.0 / input_len)
            self.linear_trend.weight.data.fill_(1.0 / input_len)

    def forward(self, past_expmap, output_len=None):
        if output_len is not None and output_len != self.output_len:
            raise ValueError("DLinear was built for output_len={}, got {}".format(self.output_len, output_len))

        seasonal_init, trend_init = self.decomp(past_expmap)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), self.feature_dim, self.output_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.feature_dim):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)


# ============================================================
# Sequence pool and synthetic bank
# ============================================================


class SequenceRealSubseriesSampler:
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

            key, subject_name, action_name, trial_name, raw_path_text = make_seq_key(idx, subject_name, action, trial, raw_path)
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
            subs.append(seq[start:start + self.synthetic_len])
        return torch.tensor(np.stack(subs), dtype=torch.float32), keys

    def get_initial_subseries(self, method="random_complete"):
        subs = []
        for info in self.seq_infos:
            key = info["key"]
            seq = self.by_key[str(key)]
            if method == "random_complete":
                motion = sample_random_concatenated(seq, self.synthetic_len, num_pieces=1)
            elif method == "random_20pieces":
                motion = sample_random_concatenated(seq, self.synthetic_len, num_pieces=20)
            else:
                raise ValueError("Unknown init method: {}".format(method))
            subs.append(motion)
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
        return {
            "seq_infos": self.seq_infos,
            "synthetic_motions": self.get_time().detach().cpu(),
        }


# ============================================================
# Segment sampling and saving
# ============================================================


def sample_random_concatenated(seq, synthetic_len, num_pieces):
    seq_len = seq.shape[0]
    if num_pieces <= 1:
        start = random.randint(0, seq_len - synthetic_len)
        return seq[start:start + synthetic_len].astype(np.float32)

    base = synthetic_len // num_pieces
    rem = synthetic_len % num_pieces
    piece_lengths = [base + (1 if i < rem else 0) for i in range(num_pieces)]
    pieces = []
    for piece_len in piece_lengths:
        if piece_len <= 0:
            continue
        start = random.randint(0, seq_len - piece_len)
        pieces.append(seq[start:start + piece_len])
    return np.concatenate(pieces, axis=0).astype(np.float32)


def save_motion_npz(path, seq_infos, motions, heldout_entries, step=0, logs=None, feature_type="expmap", feature_dim=99, tag=""):
    ensure_dir(os.path.dirname(path))

    subjects, actions, trials, lengths, raw_paths, output_motions = [], [], [], [], [], []
    for info, motion in zip(seq_infos, motions):
        subjects.append(info["subject"])
        actions.append(info["action"])
        trials.append(info["trial"])
        lengths.append(motion.shape[0])
        raw_paths.append("{}://{}".format(tag or "motion_bank", info["key"]))
        output_motions.append(np.asarray(motion, dtype=np.float32))

    for entry in heldout_entries:
        subjects.append(entry["subject"])
        actions.append(entry["action"])
        trials.append(entry["trial"])
        lengths.append(entry["length"])
        raw_paths.append(entry["raw_path"])
        output_motions.append(np.asarray(entry["motion"], dtype=np.float32))

    motion_array = np.empty(len(output_motions), dtype=object)
    for idx, motion in enumerate(output_motions):
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
        distill_logs=np.array(logs or [], dtype=object),
    )


def save_sequence_bank_npz(path, bank, step, logs, heldout_entries, feature_dim=99):
    payload = bank.get_all()
    save_motion_npz(
        path=path,
        seq_infos=payload["seq_infos"],
        motions=payload["synthetic_motions"].numpy().astype(np.float32),
        heldout_entries=heldout_entries,
        step=step,
        logs=logs,
        feature_dim=feature_dim,
        tag="distilled_step_{}".format(step),
    )


def create_random_baseline_npz(sampler, num_pieces, name, save_path, global_cfg):
    motions = []
    for info in sampler.seq_infos:
        seq = sampler.by_key[info["key"]]
        motion = sample_random_concatenated(seq, global_cfg.synthetic_len, num_pieces=num_pieces)
        motions.append(motion)
    save_motion_npz(
        save_path,
        seq_infos=sampler.seq_infos,
        motions=motions,
        heldout_entries=sampler.get_heldout_entries(),
        step=0,
        logs=[{"baseline": name, "num_pieces": num_pieces, "synthetic_len": global_cfg.synthetic_len}],
        feature_dim=global_cfg.feature_dim,
        tag=name,
    )
    print("Saved random baseline {} to {}".format(name, save_path))
    return save_path


# ============================================================
# Distillation losses
# ============================================================


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


def prediction_loss_expmap(pred_expmap, gt_future_expmap):
    return F.mse_loss(pred_expmap, gt_future_expmap)


def velocity_loss_expmap(pred_expmap, gt_future_expmap):
    pred_vel = pred_expmap[:, 1:] - pred_expmap[:, :-1]
    gt_vel = gt_future_expmap[:, 1:] - gt_future_expmap[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)


def compute_prediction_objective(backbone, past, future, output_len):
    pred = backbone(past, output_len=output_len)
    return prediction_loss_expmap(pred, future) + velocity_loss_expmap(pred, future)


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


def build_random_dlinear_backbones(input_len, output_len, feature_dim, num_backbones, device):
    backbones = []
    for _ in range(num_backbones):
        backbone = DLinearMotionBackbone(input_len, output_len, feature_dim=feature_dim).to(device)
        backbone.train()
        for param in backbone.parameters():
            param.requires_grad_(True)
        backbones.append(backbone)
    return backbones


def train_distillation(sampler, init_method, save_path, dcfg, global_cfg, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simlpe_config = build_simlpe_config()
    input_len = simlpe_config.motion.h36m_input_length
    output_len = simlpe_config.motion.h36m_target_length_train

    init_motion = sampler.get_initial_subseries(method=init_method)
    bank = SequenceFrequencySyntheticMotionBank(
        sampler.seq_infos,
        synthetic_len=global_cfg.synthetic_len,
        feature_dim=global_cfg.feature_dim,
        init_motion=init_motion,
    ).to(device)
    bank.project_valid_rfft()

    backbones = build_random_dlinear_backbones(input_len, output_len, global_cfg.feature_dim, dcfg.num_backbones, device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=dcfg.lr_synthetic)
    logs = []

    print("========== Distillation: DLinear GM ==========")
    print("Device:", device)
    print("Init method:", init_method)
    print("Num training sequences:", len(sampler.keys))
    print("Input length:", input_len, "Output length:", output_len)
    print("lambda_harm:", dcfg.lambda_harm, "lambda_grad:", dcfg.lambda_grad)
    print("lambda_vel:", dcfg.lambda_vel, "lambda_pred:", dcfg.lambda_pred)
    print("use_hdt_filter:", dcfg.use_hdt_filter, "top_k:", dcfg.top_k)
    print("Backbone: DLinear")
    print("Save path:", save_path)

    for step in range(1, dcfg.outer_steps + 1):
        real_sub_batch, keys = sampler.sample(dcfg.batch_size)
        real_sub_batch = real_sub_batch.to(device)
        syn_sub_batch = bank(keys)

        per_seq_losses, per_seq_harms, per_seq_grads, per_seq_preds, per_seq_vels, harmonic_counts = [], [], [], [], [], []
        for i in range(real_sub_batch.shape[0]):
            real_sub = real_sub_batch[i:i + 1]
            syn_sub = syn_sub_batch[i:i + 1]

            if dcfg.use_hdt_filter:
                l_harm, real_h, syn_h, harmonic_mask = harmonic_filter_and_loss(real_sub, syn_sub, top_k=dcfg.top_k)
                harmonic_count = int(harmonic_mask.sum().detach().cpu())
            else:
                l_harm = torch.zeros((), device=device)
                real_h = real_sub
                syn_h = syn_sub
                harmonic_count = 0

            real_past, real_future = make_windows_from_subseries(real_h, input_len, output_len, window_mode=dcfg.window_mode)
            syn_past, syn_future = make_windows_from_subseries(syn_h, input_len, output_len, window_mode=dcfg.window_mode)

            grad_losses = [
                compute_gradient_matching_loss_single(backbone, real_past, real_future, syn_past, syn_future, output_len)
                for backbone in backbones
            ]
            l_grad = torch.stack(grad_losses).mean()

            syn_pred = backbones[0](syn_past, output_len=output_len)
            l_pred_syn = prediction_loss_expmap(syn_pred, syn_future)
            l_vel_syn = velocity_loss_expmap(syn_pred, syn_future)

            loss_i = (
                dcfg.lambda_harm * l_harm
                + dcfg.lambda_grad * l_grad
                + dcfg.lambda_pred * l_pred_syn
                + dcfg.lambda_vel * l_vel_syn
            )
            per_seq_losses.append(loss_i)
            per_seq_harms.append(dcfg.lambda_harm * l_harm.detach())
            per_seq_grads.append(dcfg.lambda_grad * l_grad.detach())
            per_seq_preds.append(dcfg.lambda_pred * l_pred_syn.detach())
            per_seq_vels.append(dcfg.lambda_vel * l_vel_syn.detach())
            harmonic_counts.append(harmonic_count)

        total_loss = torch.stack(per_seq_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        bank.project_valid_rfft()

        with torch.no_grad():
            syn_time = bank.get_time()
            row = {
                "step": step,
                "L_harm": float(torch.stack(per_seq_harms).mean().detach().cpu()),
                "L_grad": float(torch.stack(per_seq_grads).mean().detach().cpu()),
                "L_pred_syn": float(torch.stack(per_seq_preds).mean().detach().cpu()),
                "L_vel_syn": float(torch.stack(per_seq_vels).mean().detach().cpu()),
                "L_total": float(total_loss.detach().cpu()),
                "syn_mean": float(syn_time.mean().detach().cpu()),
                "syn_std": float(syn_time.std().detach().cpu()),
                "syn_min": float(syn_time.min().detach().cpu()),
                "syn_max": float(syn_time.max().detach().cpu()),
                "harmonic_count": int(np.mean(harmonic_counts)),
                "example_key": keys[0],
            }
        logs.append(row)

        if step % dcfg.print_interval == 0:
            print(
                "distill step {step} L_harm={L_harm:.6f} L_grad={L_grad:.6f} "
                "L_pred_syn={L_pred_syn:.6f} L_vel_syn={L_vel_syn:.6f} "
                "L_total={L_total:.6f} syn_mean={syn_mean:.6f} syn_std={syn_std:.6f} "
                "syn_min={syn_min:.6f} syn_max={syn_max:.6f} example_key={example_key}".format(**row),
                flush=True,
            )

    save_sequence_bank_npz(save_path, bank, dcfg.outer_steps, logs, sampler.get_heldout_entries(), feature_dim=global_cfg.feature_dim)
    print("Saved distilled DLinear sequence-level synthetic bank to {}".format(save_path))
    return save_path


# ============================================================
# Dataset / train / eval
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


def evaluate_mpjpe_dlinear(model, npz_path, subjects, config, device):
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_eval
    train_step_len = config.motion.h36m_target_length_train
    dataset = H36MExpmapEvalDataset(npz_path, subjects, input_len, output_len)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    sums = np.zeros([output_len], dtype=np.float64)
    num_samples = 0
    model.eval()
    for motion_input_expmap, motion_target_xyz32 in dataloader:
        motion_input_expmap = motion_input_expmap.to(device)
        motion_target_xyz32 = motion_target_xyz32.float()
        batch_size = motion_input_expmap.shape[0]
        num_samples += batch_size

        outputs = []
        current_input = motion_input_expmap
        num_step = 1 if train_step_len == output_len else output_len // train_step_len + 1
        with torch.no_grad():
            for _ in range(num_step):
                pred_expmap = model(current_input, output_len=train_step_len)
                outputs.append(pred_expmap.detach().cpu())
                current_input = torch.cat([current_input[:, train_step_len:], pred_expmap], dim=1)

        pred_expmap = torch.cat(outputs, dim=1)[:, :output_len]
        motion_pred_xyz32 = expmap_to_xyz32(pred_expmap).reshape(batch_size, output_len, 32, 3) / 1000.0
        motion_gt_xyz32 = motion_target_xyz32.reshape(batch_size, output_len, 32, 3)
        mpjpe = torch.sum(torch.mean(torch.norm(motion_pred_xyz32 * 1000 - motion_gt_xyz32 * 1000, dim=3), dim=2), dim=0)
        sums += mpjpe.cpu().numpy()

    mpjpe_by_frame = sums / num_samples
    ret = {"#{}".format(idx + 1): mpjpe_by_frame[idx] for idx in range(output_len)}
    return [round(float(ret[key]), 1) for key in RESULT_KEYS]


def train_step_dlinear(model, past_expmap, future_expmap, optimizer, output_len, device):
    past_expmap = past_expmap.to(device)
    future_expmap = future_expmap.to(device)
    pred_expmap = model(past_expmap, output_len=output_len)
    loss_pred = prediction_loss_expmap(pred_expmap, future_expmap)
    loss_vel = velocity_loss_expmap(pred_expmap, future_expmap)
    loss = loss_pred + loss_vel
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_pred.item(), loss_vel.item()


def train_and_evaluate(train_npz_path, run_name, seed, sequence_pure=True):
    """
    Train/eval protocol matches the first-round experiments:

    - full data baseline:
        global shuffled DataLoader over all sliding windows.
        This matches run_dlinear_full_data_train_eval.py.

    - compressed / distilled banks:
        sequence-pure batch sampler.
        This matches the original DLinear distill/random compare script.
    """
    cfg = TrainConfig()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_simlpe_config()
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_train

    model = DLinearMotionBackbone(input_len, output_len, feature_dim=99).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_dataset = H36MExpmapWindowDataset(
        train_npz_path,
        cfg.train_subjects,
        input_len,
        output_len,
        sample_rate=cfg.sample_rate,
    )

    if sequence_pure:
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
        train_protocol = "sequence_pure"
        num_batches = len(train_batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        train_protocol = "global_shuffle_full_data"
        num_batches = len(train_loader)

    print("========== Train/Eval DLinear: {} ==========".format(run_name))
    print("Train NPZ:", train_npz_path)
    print("Train protocol:", train_protocol)
    if sequence_pure:
        print("Train batches are sequence-pure: one batch contains windows from one sequence only.")
        print("Adaptive batch sampler: batch_size <= {} num_batches = {}".format(cfg.batch_size, num_batches))
    else:
        print("Full-data protocol: global shuffled DataLoader over all sliding windows.")
        print("Batch size = {} num_batches = {}".format(cfg.batch_size, num_batches))
    print("Seed:", seed)
    print("Num train windows:", len(train_dataset))

    nb_iter, avg_loss = 0, 0.0
    while nb_iter < cfg.total_iters:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step_dlinear(model, past_expmap, future_expmap, optimizer, output_len, device)
            nb_iter += 1
            avg_loss += loss
            if nb_iter % cfg.print_every == 0:
                print("[{}] train iter {} loss={:.6f}".format(run_name, nb_iter, avg_loss / cfg.print_every), flush=True)
                avg_loss = 0.0
            if nb_iter == cfg.total_iters:
                break

    metrics = evaluate_mpjpe_dlinear(model, cfg.test_npz_path, cfg.test_subjects, config, device)
    print("[{}] Final MPJPE {}: {}".format(run_name, RESULT_KEYS, metrics))
    return metrics


# ============================================================
# Per-process runner and summary
# ============================================================


def run_one_experiment(exp):
    global_cfg = GlobalConfig()
    dcfg = DistillConfig()
    dcfg.lambda_pred = 0.0 if exp["base_name"] == "distilled_dlinear_gm_pure" else (exp["lambda_pred"] if exp["lambda_pred"] is not None else dcfg.lambda_pred)
    dcfg.lambda_vel = 0.0 if exp["base_name"] == "distilled_dlinear_gm_pure" else (exp["lambda_vel"] if exp["lambda_vel"] is not None else dcfg.lambda_vel)

    seed = int(exp["seed"])
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
            print("Base experiment:", exp["base_name"])
            print("Mode:", exp["mode"])
            print("Init method:", exp["init_method"])
            print("Use GM:", exp["use_gm"])
            print("Seed:", seed)
            print("Output NPZ:", exp_npz_path)
            print("Individual log:", exp_log_path)
            print("RESULT_KEYS:", RESULT_KEYS)
            print("=" * 80, flush=True)
            started = time.time()

            try:
                set_seed(seed)
                sampler = SequenceRealSubseriesSampler(
                    global_cfg.data_path,
                    synthetic_len=global_cfg.synthetic_len,
                    include_subjects=DISTILL_SUBJECTS,
                    heldout_subjects=HELDOUT_SUBJECTS,
                )
                print("Num training sequences:", len(sampler.keys))
                print("Synthetic length:", global_cfg.synthetic_len)

                if exp["mode"] == "full":
                    train_npz_path = global_cfg.data_path
                    print("[FULL] Using original full training data:", train_npz_path)
                elif exp["mode"] == "baseline":
                    num_pieces = 1 if exp["init_method"] == "random_complete" else 20
                    train_npz_path = create_random_baseline_npz(sampler, num_pieces, exp["base_name"], exp_npz_path, global_cfg)
                elif exp["mode"] == "gm":
                    train_npz_path = train_distillation(sampler, exp["init_method"], exp_npz_path, dcfg, global_cfg, seed)
                else:
                    raise ValueError("Unknown mode: {}".format(exp["mode"]))

                metrics = train_and_evaluate(train_npz_path, exp["name"], seed, sequence_pure=(exp["mode"] != "full"))
                elapsed = time.time() - started
                print("Elapsed seconds:", round(elapsed, 2))
                result = {
                    "name": exp["name"],
                    "base_name": exp["base_name"],
                    "seed": seed,
                    "mode": exp["mode"],
                    "init_method": exp["init_method"],
                    "use_gm": exp["use_gm"],
                    "lambda_pred": dcfg.lambda_pred if exp["mode"] == "gm" else None,
                    "lambda_vel": dcfg.lambda_vel if exp["mode"] == "gm" else None,
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
                    "base_name": exp["base_name"],
                    "seed": seed,
                    "mode": exp["mode"],
                    "init_method": exp["init_method"],
                    "use_gm": exp["use_gm"],
                    "metrics": None,
                    "log_path": exp_log_path,
                    "npz_path": exp_npz_path,
                    "status": "failed",
                    "error": repr(exc),
                }
                print("JSON_RESULT:", json.dumps(result, ensure_ascii=False))
                return result


def write_summary(summary_path, results):
    ensure_dir(os.path.dirname(summary_path))
    ordered_bases = [base["base_name"] for base in BASE_EXPERIMENTS]
    grouped = {base: [] for base in ordered_bases}
    for r in results:
        grouped.setdefault(r.get("base_name", r.get("name")), []).append(r)

    with open(summary_path, "w") as f:
        f.write("DLinear seed sweep summary for Human3.6M.\n")
        f.write("This run repeats the first-round DLinear protocols over multiple seeds: full uses global-shuffled full-data windows; compressed/distilled runs use the original sequence-pure compressed-bank protocol.\n")
        f.write("Pure GM means lambda_pred=lambda_vel=0 only for distilled_dlinear_gm_pure. Full and random baselines do not run distillation.\n")
        f.write("RESULT_KEYS: {}\n".format(RESULT_KEYS))
        f.write("Seeds: {}\n".format(list(GlobalConfig().seeds)))
        f.write("\n")

        f.write("Per-run results:\n")
        for r in sorted(results, key=lambda x: (ordered_bases.index(x.get("base_name")) if x.get("base_name") in ordered_bases else 999, int(x.get("seed") or 0))):
            f.write("{}\n".format(r.get("name")))
            f.write("  status: {}\n".format(r.get("status")))
            f.write("  base_name: {}\n".format(r.get("base_name")))
            f.write("  seed: {}\n".format(r.get("seed")))
            f.write("  mode: {}\n".format(r.get("mode")))
            f.write("  init_method: {}\n".format(r.get("init_method")))
            f.write("  use_gm: {}\n".format(r.get("use_gm")))
            f.write("  lambda_pred: {}\n".format(r.get("lambda_pred")))
            f.write("  lambda_vel: {}\n".format(r.get("lambda_vel")))
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


def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    global_cfg = GlobalConfig()
    ensure_dir(resolve_path(global_cfg.log_root))
    ensure_dir(resolve_path(global_cfg.output_dir))
    ensure_dir(os.path.dirname(resolve_path(global_cfg.summary_log)))

    summary_path = resolve_path(global_cfg.summary_log)
    experiments = build_experiments()
    print("Running {} DLinear experiments with max_workers={}".format(len(experiments), global_cfg.max_workers))
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
                    "base_name": exp["base_name"],
                    "seed": exp["seed"],
                    "mode": exp["mode"],
                    "init_method": exp["init_method"],
                    "use_gm": exp["use_gm"],
                    "metrics": None,
                    "status": "failed",
                    "error": repr(exc),
                }
            results.append(result)
            print("Finished:", result["name"], "status=", result.get("status"), "metrics=", result.get("metrics"), flush=True)
            write_summary(summary_path, results)

    ordered = {exp["name"]: i for i, exp in enumerate(experiments)}
    results = sorted(results, key=lambda r: ordered.get(r.get("name"), 999))
    write_summary(summary_path, results)
    print("Done. Summary written to", summary_path)


if __name__ == "__main__":
    main()
