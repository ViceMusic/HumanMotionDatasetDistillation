#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protocol-corrected unofficial HDT reproduction for ETT + DLinear.

This version fixes the four protocol issues from the first script:

1. ETT split is changed to the standard border split with seq_len overlap:
   ETTh: train 12 months, val 4 months, test 4 months.
   ETTm: same months, but 4x points because 15-minute frequency.

2. Full / Random / HDT final evaluation all use the same train_with_validation():
   epoch-based training, validation selection, then test with the best-val model.

3. Random / HDT no longer train for a fixed number of repeated tiny-data steps.
   They use the same epoch-based target-model protocol as Full.

4. The script prints Full first. If Full is not close to the expected DLinear ETTh1
   range, do not interpret HDT yet.

Default target line:
    Dataset: ETTh1
    Backbone: DLinear
    Eval model: DLinear
    M = 384
    input_len = 96
    pred_len = 96

Run:
    python hdt_ett_dlinear_protocol_corrected.py
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
    # Data
    data_root: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/raw/ETT"
    dataset_name: str = "ETTh1"  # ETTh1 / ETTh2 / ETTm1 / ETTm2
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # Forecasting setting in HDT paper
    input_len: int = 96
    pred_len: int = 96
    synthetic_len: int = 384

    # HDT hyperparameters
    # Paper appendix style: k in {M//4, M//8}, p=1, lr_S=0.01, inner loop=20.
    top_k: int = 384 // 4
    p_norm: int = 1
    lambda_harm: float = 1e-2
    lr_synthetic: float = 1e-2
    outer_steps: int = 1000
    validate_every: int = 50

    inner_steps_real: int = 20
    inner_steps_syn: int = 20
    inner_lr: float = 1e-3
    inner_batch_size: int = 64

    # Target-model training protocol for Full / Random / HDT.
    # Keep all three identical.
    target_epochs: int = 10
    target_batch_size: int = 64
    target_lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3

    # DLinear
    moving_avg_kernel: int = 25
    individual: bool = False

    # Reproducibility
    seed: int = 888
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output: 醒目占位，之后改成正式 repo 路径即可
    output_dir: str = "/home/user/workspace/HumanMotionDatasetDistillation/!!!HDT_DISTILLED_OUTPUT_PLACEHOLDER!!!"
    distilled_save_name: str = "hdt_ETTh1_DLinear_M384_protocol_corrected.npz"

    # Switches
    run_full: bool = True
    run_random: bool = True
    run_hdt: bool = True


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_ett_csv(path, feature_cols):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row[col]) for col in feature_cols])
    return np.asarray(rows, dtype=np.float32)


def get_ett_borders(dataset_name, seq_len):
    """
    Standard ETT border split used by Informer / Autoformer / DLinear-style code.

    ETTh:
        12 months train, 4 months val, 4 months test
        1 point per hour
    ETTm:
        same months, 4 points per hour
    """
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
    """
    Standard chronological ETT split.
    Scaler is fitted on the train segment only.
    Val/test include seq_len overlap on the left border, so the first validation/test
    window has a proper historical input.
    """
    border1s, border2s = get_ett_borders(cfg.dataset_name, cfg.input_len)

    if border2s[-1] > len(values):
        raise ValueError(
            "Dataset length {} is shorter than expected border {} for {}".format(
                len(values), border2s[-1], cfg.dataset_name
            )
        )

    train_raw_for_scaler = values[border1s[0]:border2s[0]]
    mean = train_raw_for_scaler.mean(axis=0, keepdims=True)
    std = train_raw_for_scaler.std(axis=0, keepdims=True) + 1e-6

    values_norm = (values - mean) / std

    train = values_norm[border1s[0]:border2s[0]].astype(np.float32)
    val = values_norm[border1s[1]:border2s[1]].astype(np.float32)
    test = values_norm[border1s[2]:border2s[2]].astype(np.float32)

    return train, val, test, mean.astype(np.float32), std.astype(np.float32), border1s, border2s


def make_all_windows(series, input_len, pred_len):
    """
    series: numpy or torch [N, C]
    return:
        x: [num_windows, input_len, C]
        y: [num_windows, pred_len, C]
    """
    if isinstance(series, np.ndarray):
        series = torch.from_numpy(series).float()

    total = input_len + pred_len
    n = series.shape[0]
    if n < total:
        raise ValueError("Series length {} < input_len + pred_len {}".format(n, total))

    xs = []
    ys = []
    for start in range(n - total + 1):
        window = series[start:start + total]
        xs.append(window[:input_len])
        ys.append(window[input_len:])

    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def sample_contiguous_subsequence(series_np, length):
    n = series_np.shape[0]
    if n < length:
        raise ValueError("Cannot sample length {} from series length {}".format(length, n))
    start = random.randint(0, n - length)
    return torch.from_numpy(series_np[start:start + length]).float()


def iterate_minibatches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def sample_window_batch(x, y, batch_size, device):
    n = x.shape[0]
    if n <= batch_size:
        idx = torch.arange(n, device=x.device)
    else:
        idx = torch.randint(0, n, (batch_size,), device=x.device)
    return x[idx].to(device), y[idx].to(device)


# ============================================================
# DLinear
# ============================================================

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_avg = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_avg


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
        # x: [B, L, C]
        seasonal, trend = self.decomp(x)

        seasonal = seasonal.permute(0, 2, 1)  # [B, C, L]
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = []
            trend_out = []
            for c in range(self.channels):
                seasonal_out.append(self.linear_seasonal[c](seasonal[:, c, :]))
                trend_out.append(self.linear_trend[c](trend[:, c, :]))
            seasonal_out = torch.stack(seasonal_out, dim=1)
            trend_out = torch.stack(trend_out, dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        out = seasonal_out + trend_out
        return out.permute(0, 2, 1)  # [B, T, C]


def build_dlinear(cfg, channels):
    return DLinear(
        seq_len=cfg.input_len,
        pred_len=cfg.pred_len,
        channels=channels,
        moving_avg_kernel=cfg.moving_avg_kernel,
        individual=cfg.individual,
    )


# ============================================================
# Full / Random / HDT target-model training protocol
# ============================================================

def evaluate_windows(model, x, y, cfg, desc=None):
    device = cfg.device
    model.eval()

    x = x.to(device)
    y = y.to(device)

    total = 0.0
    count = 0

    with torch.no_grad():
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
    """
    Shared target-model protocol for Full / Random / HDT:
        train by epochs
        evaluate on validation each epoch
        keep best validation checkpoint
        return best model and best val mse

    train_series / val_series: numpy or torch [N, C]
    """
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
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_mse = evaluate_windows(model, x_val, y_val, cfg)
        train_loss = float(np.mean(losses)) if losses else float("nan")

        print(
            "[{}] epoch {}/{} train_mse={:.6f} val_mse={:.6f}".format(
                desc, epoch, cfg.target_epochs, train_loss, val_mse
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


# ============================================================
# HDT frequency synthetic series
# ============================================================

class FrequencySyntheticSeries(nn.Module):
    def __init__(self, init_series):
        super().__init__()
        init_freq = torch.fft.rfft(init_series, dim=0)
        self.M = init_series.shape[0]
        self.C = init_series.shape[1]
        self.freq_real = nn.Parameter(init_freq.real.clone())
        self.freq_imag = nn.Parameter(init_freq.imag.clone())

    def freq(self):
        return torch.complex(self.freq_real, self.freq_imag)

    def time(self):
        return torch.fft.irfft(self.freq(), n=self.M, dim=0)

    @torch.no_grad()
    def project_valid_rfft(self):
        self.freq_imag[0, :].zero_()
        if self.M % 2 == 0:
            self.freq_imag[-1, :].zero_()


def build_channelwise_topk_mask(F_X, top_k):
    """
    Channel-independent top-k harmonics:
        for each variable/channel, select top-k frequency bins according to |F_X|.
    """
    amp = torch.abs(F_X.detach())  # [F, C]
    f_bins, c = amp.shape
    k = min(top_k, f_bins)

    idx = torch.topk(amp, k=k, dim=0).indices  # [k, C]
    mask = torch.zeros_like(amp, dtype=torch.bool)
    channel_ids = torch.arange(c, device=F_X.device).view(1, c).expand(k, c)
    mask[idx, channel_ids] = True
    return mask


def harmonic_step(X_sub, synthetic, cfg):
    """
    Paper-style harmonic step:
        F_X = FFT(X_sub)
        F_S = current synthetic frequency
        H = top-k from |F_X|
        L_harm = || |F_X^H| - |F_S^H| ||_p
        X_H = iFFT(F_X^H)
        S_H = iFFT(F_S^H)
    """
    F_X = torch.fft.rfft(X_sub, dim=0)
    F_S = synthetic.freq()

    mask = build_channelwise_topk_mask(F_X, cfg.top_k)

    F_X_h = torch.where(mask, F_X, torch.zeros_like(F_X))
    F_S_h = torch.where(mask, F_S, torch.zeros_like(F_S))

    if cfg.p_norm == 1:
        L_harm = torch.mean(torch.abs(torch.abs(F_X_h.detach()) - torch.abs(F_S_h)))
    elif cfg.p_norm == 2:
        L_harm = torch.mean((torch.abs(F_X_h.detach()) - torch.abs(F_S_h)) ** 2)
    else:
        raise ValueError("Unsupported p_norm: {}".format(cfg.p_norm))

    X_H = torch.fft.irfft(F_X_h, n=cfg.synthetic_len, dim=0)
    S_H = torch.fft.irfft(F_S_h, n=cfg.synthetic_len, dim=0)

    return L_harm, X_H, S_H


# ============================================================
# Differentiable trajectory matching Eq.12-style
# ============================================================

def clone_params(model, detach=True, requires_grad=True):
    params = {}
    for name, p in model.named_parameters():
        q = p.detach().clone() if detach else p.clone()
        q.requires_grad_(requires_grad)
        params[name] = q
    return params


def get_buffers(model):
    return {name: b for name, b in model.named_buffers()}


def functional_forward(model, params, buffers, x):
    state = {}
    state.update(params)
    state.update(buffers)

    try:
        from torch.func import functional_call
        return functional_call(model, state, (x,))
    except Exception:
        from torch.nn.utils.stateless import functional_call
        return functional_call(model, state, (x,))


def inner_update(model, params, buffers, x, y, lr, create_graph):
    pred = functional_forward(model, params, buffers, x)
    loss = F.mse_loss(pred, y)

    names = list(params.keys())
    values = [params[n] for n in names]
    grads = torch.autograd.grad(
        loss,
        values,
        create_graph=create_graph,
        allow_unused=True,
    )

    new_params = {}
    for n, p, g in zip(names, values, grads):
        new_params[n] = p if g is None else p - lr * g

    return new_params, loss


def trajectory_matching_loss(model, X_H, S_H, cfg):
    """
    L_grad = ||T_j(theta,S_H)-T_i(theta,X_H)||^2 /
             ||theta-T_i(theta,X_H)||^2
    """
    device = cfg.device

    x_real, y_real = make_all_windows(X_H, cfg.input_len, cfg.pred_len)
    x_syn, y_syn = make_all_windows(S_H, cfg.input_len, cfg.pred_len)

    x_real = x_real.to(device)
    y_real = y_real.to(device)
    x_syn = x_syn.to(device)
    y_syn = y_syn.to(device)

    base_params = clone_params(model, detach=True, requires_grad=False)
    buffers = get_buffers(model)

    real_params = clone_params(model, detach=True, requires_grad=True)
    syn_params = clone_params(model, detach=True, requires_grad=True)

    # Real trajectory: target trajectory, detached after each update.
    for _ in range(cfg.inner_steps_real):
        xb, yb = sample_window_batch(x_real.detach(), y_real.detach(), cfg.inner_batch_size, device)
        real_params, _ = inner_update(
            model, real_params, buffers, xb, yb,
            lr=cfg.inner_lr,
            create_graph=False,
        )
        real_params = {n: p.detach().requires_grad_(True) for n, p in real_params.items()}

    # Synthetic trajectory: keep graph to S_H/F_S.
    for _ in range(cfg.inner_steps_syn):
        xb, yb = sample_window_batch(x_syn, y_syn, cfg.inner_batch_size, device)
        syn_params, _ = inner_update(
            model, syn_params, buffers, xb, yb,
            lr=cfg.inner_lr,
            create_graph=True,
        )

    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)

    for name in base_params.keys():
        theta0 = base_params[name].detach()
        theta_real = real_params[name].detach()
        theta_syn = syn_params[name]

        numerator = numerator + (theta_syn - theta_real).pow(2).sum()
        denominator = denominator + (theta0 - theta_real).pow(2).sum()

    return numerator / (denominator + 1e-8)


# ============================================================
# HDT distillation loop
# ============================================================

def run_hdt_distillation(train_np, val_np, cfg, channels):
    device = cfg.device

    init_S = sample_contiguous_subsequence(train_np, cfg.synthetic_len).to(device)
    synthetic = FrequencySyntheticSeries(init_S).to(device)
    synthetic.project_valid_rfft()

    optimizer = torch.optim.Adam(synthetic.parameters(), lr=cfg.lr_synthetic)

    best_val = float("inf")
    best_S = None

    print("\n========== HDT Distillation ==========")
    print("M={}, top_k={}, p={}, lambda_harm={}, lr_S={}".format(
        cfg.synthetic_len, cfg.top_k, cfg.p_norm, cfg.lambda_harm, cfg.lr_synthetic
    ))
    print("outer_steps={}, inner_steps_real={}, inner_steps_syn={}".format(
        cfg.outer_steps, cfg.inner_steps_real, cfg.inner_steps_syn
    ))

    for outer in range(1, cfg.outer_steps + 1):
        # Paper Algorithm 1 initializes theta each outer loop.
        model = build_dlinear(cfg, channels).to(device)
        model.train()

        X_sub = sample_contiguous_subsequence(train_np, cfg.synthetic_len).to(device)

        L_harm, X_H, S_H = harmonic_step(X_sub, synthetic, cfg)
        L_grad = trajectory_matching_loss(model, X_H, S_H, cfg)

        loss = L_grad + cfg.lambda_harm * L_harm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        synthetic.project_valid_rfft()

        if outer % 10 == 0 or outer == 1:
            with torch.no_grad():
                S_now = synthetic.time()
                print(
                    "[HDT] outer {}/{} L_grad={:.6f} L_harm_w={:.6f} L_total={:.6f} "
                    "S_mean={:.4f} S_std={:.4f}".format(
                        outer,
                        cfg.outer_steps,
                        L_grad.item(),
                        (cfg.lambda_harm * L_harm).item(),
                        loss.item(),
                        S_now.mean().item(),
                        S_now.std().item(),
                    )
                )

        # Paper reports validation every 50 outer epochs.
        if outer % cfg.validate_every == 0 or outer == cfg.outer_steps:
            with torch.no_grad():
                S_eval = synthetic.time().detach().cpu()

            val_model, val_mse = train_with_validation(
                S_eval,
                val_np,
                cfg,
                channels,
                desc="HDT-target@outer{}".format(outer),
            )

            if val_mse < best_val:
                best_val = val_mse
                best_S = S_eval.clone()
                print("[HDT] New best S at outer {}, val_mse={:.6f}".format(outer, best_val))

    if best_S is None:
        best_S = synthetic.time().detach().cpu()

    return best_S, best_val


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)

    data_path = os.path.join(cfg.data_root, "{}.csv".format(cfg.dataset_name))
    save_path = os.path.join(cfg.output_dir, cfg.distilled_save_name)

    ensure_dir(cfg.output_dir)

    print("========== Protocol-corrected HDT ETT DLinear ==========")
    print("Data:", data_path)
    print("Save path:", save_path)
    print("Device:", cfg.device)

    values = load_ett_csv(data_path, cfg.feature_cols)
    train_np, val_np, test_np, mean, std, border1s, border2s = split_ett_standard_and_normalize(values, cfg)
    channels = train_np.shape[1]

    print("Raw values:", values.shape)
    print("Borders:", border1s, border2s)
    print("Train/Val/Test series:", train_np.shape, val_np.shape, test_np.shape)
    print("Windows:")
    print("  train:", make_all_windows(train_np, cfg.input_len, cfg.pred_len)[0].shape[0])
    print("  val:  ", make_all_windows(val_np, cfg.input_len, cfg.pred_len)[0].shape[0])
    print("  test: ", make_all_windows(test_np, cfg.input_len, cfg.pred_len)[0].shape[0])

    results = {}

    if cfg.run_full:
        print("\n========== Full Data Baseline ==========")
        full_model, full_val = train_with_validation(
            train_np,
            val_np,
            cfg,
            channels,
            desc="Full",
        )
        full_test = evaluate_windows(
            full_model,
            *make_all_windows(test_np, cfg.input_len, cfg.pred_len),
            cfg=cfg,
            desc="Full-Test",
        )
        results["full_val"] = full_val
        results["full_test"] = full_test

    if cfg.run_random:
        print("\n========== Random Subsequence Baseline ==========")
        random_S = sample_contiguous_subsequence(train_np, cfg.synthetic_len)
        random_model, random_val = train_with_validation(
            random_S,
            val_np,
            cfg,
            channels,
            desc="Random",
        )
        random_test = evaluate_windows(
            random_model,
            *make_all_windows(test_np, cfg.input_len, cfg.pred_len),
            cfg=cfg,
            desc="Random-Test",
        )
        results["random_val"] = random_val
        results["random_test"] = random_test

    if cfg.run_hdt:
        best_S, best_val = run_hdt_distillation(train_np, val_np, cfg, channels)
        results["hdt_best_val_during_distill"] = best_val

        np.savez(
            save_path,
            S=best_S.numpy().astype(np.float32),
            dataset_name=np.array(cfg.dataset_name, dtype=object),
            M=np.array(cfg.synthetic_len, dtype=np.int64),
            input_len=np.array(cfg.input_len, dtype=np.int64),
            pred_len=np.array(cfg.pred_len, dtype=np.int64),
            top_k=np.array(cfg.top_k, dtype=np.int64),
            p_norm=np.array(cfg.p_norm, dtype=np.int64),
            lambda_harm=np.array(cfg.lambda_harm, dtype=np.float32),
            mean=mean,
            std=std,
            feature_cols=np.array(cfg.feature_cols, dtype=object),
            border1s=np.array(border1s, dtype=np.int64),
            border2s=np.array(border2s, dtype=np.int64),
        )
        print("[HDT] Saved distilled S to:", save_path)

        print("\n========== Final Fresh DLinear on Best HDT S ==========")
        hdt_model, hdt_val = train_with_validation(
            best_S,
            val_np,
            cfg,
            channels,
            desc="HDT-final",
        )
        hdt_test = evaluate_windows(
            hdt_model,
            *make_all_windows(test_np, cfg.input_len, cfg.pred_len),
            cfg=cfg,
            desc="HDT-Test",
        )
        results["hdt_val"] = hdt_val
        results["hdt_test"] = hdt_test

    print("\n========== Summary ==========")
    for k, v in results.items():
        print("{}: {:.6f}".format(k, v))


if __name__ == "__main__":
    main()
