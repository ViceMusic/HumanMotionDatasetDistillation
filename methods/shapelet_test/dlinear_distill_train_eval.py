import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backbones.simlpe import (
    build_simlpe_config,
    expmap_to_xyz32,
)


# ============================================================
# Full-data DLinear motion forecasting baseline
#
# Save as:
#   methods/smoke_test/run_dlinear_full_data_train_eval.py
#
# Run:
#   python methods/smoke_test/run_dlinear_full_data_train_eval.py
#
# What this script does:
#   1. No distillation.
#   2. No random compressed bank.
#   3. Train DLinear directly on the original full Human3.6M training data.
#   4. Evaluate on S5 with the same MPJPE frames used in the other scripts.
# ============================================================


TRAIN_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
TEST_SUBJECTS = ["S5"]
RESULT_KEYS = ["#2", "#4", "#8", "#10", "#14", "#18", "#22", "#25"]


@dataclass
class TrainConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    log_path: str = "logs/train_eval_dlinear_full_data.txt"

    train_subjects = TRAIN_SUBJECTS
    test_subjects = TEST_SUBJECTS

    sample_rate: int = 1
    shift_step: int = 1
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    total_iters: int = 8000
    print_every: int = 100
    seed: int = 888
    lr: float = 3e-4
    weight_decay: float = 1e-4

    dlinear_moving_avg: int = 25
    dlinear_individual: bool = False


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


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, T, C]
        if self.kernel_size <= 1:
            return x
        pad_len = (self.kernel_size - 1) // 2
        pad_front = x[:, 0:1, :].repeat(1, pad_len, 1)
        pad_end = x[:, -1:, :].repeat(1, pad_len, 1)
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
    """
    Plain DLinear for Human3.6M expmap forecasting.

    Input:
        past_expmap: [B, input_len, 99]
    Output:
        pred_expmap: [B, output_len, 99]
    """

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
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, T]
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

        return (seasonal_output + trend_output).permute(0, 2, 1)  # [B, output_len, C]


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
            raise ValueError("No windows found in {} for subjects {}".format(npz_path, subjects))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        seq_idx, start = self.data_idx[index]
        window = self.seqs[seq_idx][start : start + self.total_len]
        return window[: self.input_len], window[self.input_len :]


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


def prediction_loss_expmap(pred_expmap, gt_future_expmap):
    return F.mse_loss(pred_expmap, gt_future_expmap)


def velocity_loss_expmap(pred_expmap, gt_future_expmap):
    pred_vel = pred_expmap[:, 1:] - pred_expmap[:, :-1]
    gt_vel = gt_future_expmap[:, 1:] - gt_future_expmap[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)


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


def evaluate_mpjpe_dlinear(model, npz_path, subjects, config, cfg, device):
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_eval
    train_step_len = config.motion.h36m_target_length_train

    dataset = H36MExpmapEvalDataset(
        npz_path,
        subjects,
        input_len,
        output_len,
        shift_step=cfg.shift_step,
        sample_rate=cfg.sample_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
    )

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

        mpjpe = torch.sum(
            torch.mean(torch.norm(motion_pred_xyz32 * 1000 - motion_gt_xyz32 * 1000, dim=3), dim=2),
            dim=0,
        )
        sums += mpjpe.cpu().numpy()

    mpjpe_by_frame = sums / num_samples
    ret = {}
    for idx in range(output_len):
        ret["#{}".format(idx + 1)] = mpjpe_by_frame[idx]
    return [round(ret[key], 1) for key in RESULT_KEYS]


def main():
    cfg = TrainConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_simlpe_config()
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_train

    train_dataset = H36MExpmapWindowDataset(
        cfg.data_path,
        cfg.train_subjects,
        input_len,
        output_len,
        shift_step=cfg.shift_step,
        sample_rate=cfg.sample_rate,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = DLinearMotionBackbone(
        input_len,
        output_len,
        feature_dim=99,
        moving_avg=cfg.dlinear_moving_avg,
        individual=cfg.dlinear_individual,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path = resolve_path(cfg.log_path)
    ensure_dir(os.path.dirname(log_path))
    acc_log = open(log_path, "w")
    acc_log.write("Run : dlinear_full_data\n")
    acc_log.write("Seed : {}\n".format(cfg.seed))
    acc_log.write("Backbone : DLinear\n")
    acc_log.write("Train NPZ : {}\n".format(cfg.data_path))
    acc_log.write("Test NPZ : {}\n".format(cfg.data_path))
    acc_log.write("Train subjects : {}\n".format(" ".join(cfg.train_subjects)))
    acc_log.write("Test subjects : {}\n".format(" ".join(cfg.test_subjects)))
    acc_log.write("Input length : {}\n".format(input_len))
    acc_log.write("Train output length : {}\n".format(output_len))
    acc_log.write("Total train windows : {}\n".format(len(train_dataset)))
    acc_log.write("Total train batches per epoch-like pass : {}\n".format(len(train_loader)))

    print("========== DLinear Full Data Train/Eval ==========")
    print("Device:", device)
    print("Data path:", cfg.data_path)
    print("Train subjects:", cfg.train_subjects)
    print("Test subjects:", cfg.test_subjects)
    print("Input length:", input_len, "Train output length:", output_len)
    print("Total train windows:", len(train_dataset))
    print("Batch size:", cfg.batch_size, "Total iters:", cfg.total_iters)
    print("Training uses the original full dataset windows, not distilled/random compressed sequences.")

    nb_iter = 0
    avg_loss = 0.0
    avg_pred = 0.0
    avg_vel = 0.0

    while nb_iter < cfg.total_iters:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step_dlinear(model, past_expmap, future_expmap, optimizer, output_len, device)
            nb_iter += 1
            avg_loss += loss
            avg_pred += loss_pred
            avg_vel += loss_vel

            if nb_iter % cfg.print_every == 0:
                msg = "train iter {} loss={:.6f} pred={:.6f} vel={:.6f}".format(
                    nb_iter,
                    avg_loss / cfg.print_every,
                    avg_pred / cfg.print_every,
                    avg_vel / cfg.print_every,
                )
                print(msg)
                acc_log.write(msg + "\n")
                acc_log.flush()
                avg_loss = 0.0
                avg_pred = 0.0
                avg_vel = 0.0

            if nb_iter >= cfg.total_iters:
                break

    metrics = evaluate_mpjpe_dlinear(model, cfg.data_path, cfg.test_subjects, config, cfg, device)
    print("Final MPJPE {}: {}".format(RESULT_KEYS, metrics))
    acc_log.write("final\n{}\n".format(" ".join(str(v) for v in metrics)))
    acc_log.flush()
    acc_log.close()

    print("Saved train/eval log to {}".format(log_path))


if __name__ == "__main__":
    main()
