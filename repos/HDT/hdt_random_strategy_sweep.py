#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random strategy diagnostic for HDT / ETT / DLinear.

Purpose:
    Test why the Random baseline is strong/weak by comparing several definitions:

    1. contiguous:
        one continuous real subsequence of length M.

    2. concat_piece_K:
        sample K short continuous pieces from random positions, concatenate to length M.
        K=16,32,64,128,384.  K=384 means each piece length is 1.

    3. random_points_unsorted:
        sample M individual time points independently from train, keep random sampled order.

    4. random_points_sorted:
        sample M individual time points from train, sort by original time index.

All variants use exactly the same target-model training protocol:
    DLinear
    input_len = 96
    pred_len = 96
    M = 384
    standard ETT split
    train with validation
    evaluate best-val checkpoint on test

Run:
    python hdt_random_strategy_sweep.py

No HDT distillation is run in this file. This is only for Random baseline diagnosis.
"""

import os
import csv
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    data_root: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/raw/ETT"
    dataset_name: str = "ETTh1"
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    input_len: int = 96
    pred_len: int = 96
    synthetic_len: int = 384

    # Random strategies to test
    concat_piece_counts = [16, 32, 64, 128, 384]
    num_random_trials: int = 3

    # Target-model protocol
    target_epochs: int = 10
    target_batch_size: int = 64
    target_lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3

    # DLinear
    moving_avg_kernel: int = 25
    individual: bool = False

    # Reproducibility/device
    seed: int = 888
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Switches
    run_full: bool = True
    run_contiguous: bool = True
    run_concat_pieces: bool = True
    run_random_points_unsorted: bool = True
    run_random_points_sorted: bool = True


# ============================================================
# Utils
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ett_csv(path, feature_cols):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[col]) for col in feature_cols])
    return np.asarray(rows, dtype=np.float32)


def get_ett_borders(dataset_name, seq_len):
    if dataset_name in ["ETTh1", "ETTh2"]:
        unit = 30 * 24
    elif dataset_name in ["ETTm1", "ETTm2"]:
        unit = 30 * 24 * 4
    else:
        raise ValueError("Unknown ETT dataset_name: {}".format(dataset_name))

    border1s = [
        0,
        12 * unit - seq_len,
        12 * unit + 4 * unit - seq_len,
    ]
    border2s = [
        12 * unit,
        12 * unit + 4 * unit,
        12 * unit + 8 * unit,
    ]
    return border1s, border2s


def split_ett_standard_and_normalize(values, cfg):
    border1s, border2s = get_ett_borders(cfg.dataset_name, cfg.input_len)

    if border2s[-1] > len(values):
        raise ValueError(
            "Dataset length {} shorter than expected border {} for {}".format(
                len(values), border2s[-1], cfg.dataset_name
            )
        )

    train_raw = values[border1s[0]:border2s[0]]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-6

    values_norm = (values - mean) / std

    train = values_norm[border1s[0]:border2s[0]].astype(np.float32)
    val = values_norm[border1s[1]:border2s[1]].astype(np.float32)
    test = values_norm[border1s[2]:border2s[2]].astype(np.float32)

    return train, val, test, mean.astype(np.float32), std.astype(np.float32), border1s, border2s


def make_all_windows(series, input_len, pred_len):
    if isinstance(series, np.ndarray):
        series = torch.from_numpy(series).float()

    total = input_len + pred_len
    n = series.shape[0]
    if n < total:
        raise ValueError("Series length {} < {}".format(n, total))

    xs, ys = [], []
    for start in range(n - total + 1):
        window = series[start:start + total]
        xs.append(window[:input_len])
        ys.append(window[input_len:])
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def iterate_minibatches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


# ============================================================
# Random data constructors
# ============================================================

def random_contiguous(series_np, M):
    n = series_np.shape[0]
    start = random.randint(0, n - M)
    S = series_np[start:start + M]
    meta = {"start": start}
    return torch.from_numpy(S.astype(np.float32)).float(), meta


def random_concat_pieces(series_np, M, num_pieces):
    """
    Sample num_pieces short continuous pieces from random positions,
    concatenate them to length M.

    num_pieces=384 means piece_len=1, equivalent to random individual points
    in sampled order.
    """
    n = series_np.shape[0]
    piece_len = int(np.ceil(M / num_pieces))
    if n < piece_len:
        raise ValueError("n {} < piece_len {}".format(n, piece_len))

    pieces = []
    starts = []
    for _ in range(num_pieces):
        s = random.randint(0, n - piece_len)
        starts.append(s)
        pieces.append(series_np[s:s + piece_len])

    S = np.concatenate(pieces, axis=0)[:M]
    meta = {"num_pieces": num_pieces, "piece_len": piece_len, "starts_head": starts[:10]}
    return torch.from_numpy(S.astype(np.float32)).float(), meta


def random_points_unsorted(series_np, M):
    """
    Sample M individual time points from random positions.
    Keep sampled random order.
    """
    n = series_np.shape[0]
    idx = np.random.choice(n, size=M, replace=False)
    S = series_np[idx]
    meta = {"idx_head": idx[:10].tolist()}
    return torch.from_numpy(S.astype(np.float32)).float(), meta


def random_points_sorted(series_np, M):
    """
    Sample M individual time points from random positions.
    Sort by original time index.
    This preserves chronological order but not uniform time interval or continuity.
    """
    n = series_np.shape[0]
    idx = np.random.choice(n, size=M, replace=False)
    idx = np.sort(idx)
    S = series_np[idx]
    meta = {"idx_head": idx[:10].tolist()}
    return torch.from_numpy(S.astype(np.float32)).float(), meta


# ============================================================
# DLinear
# ============================================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels, moving_avg_kernel=25, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg_kernel)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

        self.init_weights()

    def init_weights(self):
        if self.individual:
            for c in range(self.channels):
                nn.init.constant_(self.linear_seasonal[c].weight, 1.0 / self.seq_len)
                nn.init.constant_(self.linear_trend[c].weight, 1.0 / self.seq_len)
                nn.init.zeros_(self.linear_seasonal[c].bias)
                nn.init.zeros_(self.linear_trend[c].bias)
        else:
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
            nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.linear_seasonal.bias)
            nn.init.zeros_(self.linear_trend.bias)

    def forward(self, x):
        seasonal, trend = self.decomp(x)

        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out, trend_out = [], []
            for c in range(self.channels):
                seasonal_out.append(self.linear_seasonal[c](seasonal[:, c, :]))
                trend_out.append(self.linear_trend[c](trend[:, c, :]))
            seasonal_out = torch.stack(seasonal_out, dim=1)
            trend_out = torch.stack(trend_out, dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        return (seasonal_out + trend_out).permute(0, 2, 1)


def build_dlinear(cfg, channels):
    return DLinear(
        seq_len=cfg.input_len,
        pred_len=cfg.pred_len,
        channels=channels,
        moving_avg_kernel=cfg.moving_avg_kernel,
        individual=cfg.individual,
    )


# ============================================================
# Train / eval
# ============================================================

@torch.no_grad()
def evaluate_windows(model, x, y, cfg, desc=None):
    device = cfg.device
    model.eval()

    x = x.to(device)
    y = y.to(device)

    total = 0.0
    count = 0

    for xb, yb in iterate_minibatches(x, y, batch_size=256, shuffle=False):
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        total += F.mse_loss(pred, yb, reduction="sum").item()
        count += yb.numel()

    mse = total / count
    if desc is not None:
        print("[{}] MSE={:.6f}".format(desc, mse))
    return mse


def train_with_validation(train_series, val_series, cfg, channels, desc):
    device = cfg.device

    x_train, y_train = make_all_windows(train_series, cfg.input_len, cfg.pred_len)
    x_val, y_val = make_all_windows(val_series, cfg.input_len, cfg.pred_len)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    model = build_dlinear(cfg, channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.target_lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.target_epochs + 1):
        model.train()
        losses = []

        for xb, yb in iterate_minibatches(x_train, y_train, cfg.target_batch_size, shuffle=True):
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_mse = evaluate_windows(model, x_val, y_val, cfg)
        train_mse = float(np.mean(losses)) if losses else float("nan")

        print(
            "[{}] epoch {}/{} train_mse={:.6f} val_mse={:.6f}".format(
                desc, epoch, cfg.target_epochs, train_mse, val_mse
            )
        )

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print("[{}] early stop at epoch {}, best_val={:.6f}".format(desc, epoch, best_val))
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def run_one_strategy(name, constructor, train_np, val_np, test_np, cfg, channels, trial):
    S, meta = constructor(train_np, cfg.synthetic_len)

    print("\n========== {} / trial {} ==========".format(name, trial))
    print("meta:", meta)
    print("S shape:", tuple(S.shape))

    model, val_mse = train_with_validation(S, val_np, cfg, channels, desc="{}-trial{}".format(name, trial))

    x_test, y_test = make_all_windows(test_np, cfg.input_len, cfg.pred_len)
    test_mse = evaluate_windows(model, x_test, y_test, cfg, desc="{}-trial{}-Test".format(name, trial))

    return val_mse, test_mse


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)

    data_path = os.path.join(cfg.data_root, "{}.csv".format(cfg.dataset_name))

    print("========== Random Strategy Sweep for HDT ==========")
    print("Data:", data_path)
    print("Device:", cfg.device)
    print("M:", cfg.synthetic_len)
    print("input_len/pred_len:", cfg.input_len, cfg.pred_len)

    values = load_ett_csv(data_path, cfg.feature_cols)
    train_np, val_np, test_np, mean, std, border1s, border2s = split_ett_standard_and_normalize(values, cfg)
    channels = train_np.shape[1]

    print("Raw:", values.shape)
    print("Borders:", border1s, border2s)
    print("Train/Val/Test:", train_np.shape, val_np.shape, test_np.shape)
    print("Windows train/val/test:",
          make_all_windows(train_np, cfg.input_len, cfg.pred_len)[0].shape[0],
          make_all_windows(val_np, cfg.input_len, cfg.pred_len)[0].shape[0],
          make_all_windows(test_np, cfg.input_len, cfg.pred_len)[0].shape[0])

    results = {}

    if cfg.run_full:
        print("\n========== Full Data Baseline ==========")
        full_model, full_val = train_with_validation(train_np, val_np, cfg, channels, desc="Full")
        x_test, y_test = make_all_windows(test_np, cfg.input_len, cfg.pred_len)
        full_test = evaluate_windows(full_model, x_test, y_test, cfg, desc="Full-Test")
        results["full"] = [(full_val, full_test)]

    if cfg.run_contiguous:
        vals = []
        for trial in range(1, cfg.num_random_trials + 1):
            vals.append(run_one_strategy("random_contiguous", random_contiguous, train_np, val_np, test_np, cfg, channels, trial))
        results["random_contiguous"] = vals

    if cfg.run_concat_pieces:
        for k in cfg.concat_piece_counts:
            def ctor(series_np, M, kk=k):
                return random_concat_pieces(series_np, M, kk)
            name = "random_concat_piece_{}".format(k)
            vals = []
            for trial in range(1, cfg.num_random_trials + 1):
                vals.append(run_one_strategy(name, ctor, train_np, val_np, test_np, cfg, channels, trial))
            results[name] = vals

    if cfg.run_random_points_unsorted:
        vals = []
        for trial in range(1, cfg.num_random_trials + 1):
            vals.append(run_one_strategy("random_points_unsorted", random_points_unsorted, train_np, val_np, test_np, cfg, channels, trial))
        results["random_points_unsorted"] = vals

    if cfg.run_random_points_sorted:
        vals = []
        for trial in range(1, cfg.num_random_trials + 1):
            vals.append(run_one_strategy("random_points_sorted", random_points_sorted, train_np, val_np, test_np, cfg, channels, trial))
        results["random_points_sorted"] = vals

    print("\n========== Summary ==========")
    for name, vals in results.items():
        arr = np.asarray(vals, dtype=np.float64)
        val_mean = arr[:, 0].mean()
        val_std = arr[:, 0].std()
        test_mean = arr[:, 1].mean()
        test_std = arr[:, 1].std()

        print(
            "{}: val={:.6f}±{:.6f}, test={:.6f}±{:.6f}, trials={}".format(
                name,
                val_mean,
                val_std,
                test_mean,
                test_std,
                ["{:.6f}".format(x[1]) for x in vals],
            )
        )


if __name__ == "__main__":
    main()
