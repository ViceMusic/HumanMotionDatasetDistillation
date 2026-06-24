#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from backbones.simlpe import (
    SiMLPeMotionBackbone,
    build_simlpe_config,
    expmap_to_simlpe_xyz66,
    expmap_to_xyz32,
)

# ============================================================
# Random sequence construction + train/eval for Human3.6M motion
# Save as:
#   methods/smoke_test/run_random_position_sequence_methods_and_train.py
#
# Run:
#   python methods/smoke_test/run_random_position_sequence_methods_and_train.py
#
# This script DOES NOT run HDT / distillation.
# It saves NPZ datasets with the same format as your original processed file,
# then trains/evaluates SiMLPe on each random dataset.
# ============================================================

DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
HELDOUT_SUBJECTS = ["S5"]
TEST_SUBJECTS = ["S5"]


@dataclass
class RandomBuildConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"
    synthetic_len: int = 100
    feature_dim: int = 99
    seed: int = 888
    methods = (
        # These four are the sanity-check baselines for the "which 100 frames?" question.
        # NOTE: the old "contiguous_100" baseline is exactly per_sequence_random100:
        # for every training sequence, sample one random contiguous 100-frame crop.
        "per_sequence_first100",
        "per_sequence_middle100",
        "per_sequence_last100",
        "per_sequence_random100",

        # Keep the previous fragmented/random baselines for comparison.
        "piece_10_nonoverlap_chrono",
        "piece_20_nonoverlap_chrono",
        "piece_50_nonoverlap_chrono",
        "points_100_no_adjacent_chrono",
    )
    save_prefix: str = "h36m_expmap_sequences_random_position_len100"


@dataclass
class TrainConfig:
    train_npz_path: str = ""
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    log_dir: str = "logs"
    train_subjects = DISTILL_SUBJECTS
    test_subjects = TEST_SUBJECTS
    sample_rate: int = 1
    batch_size: int = 16
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


# ============================================================
# Random sampling methods
# ============================================================

def sample_first_contiguous(seq, out_len):
    start = 0
    return seq[start:start + out_len].copy(), {
        "start": start,
        "mode": "per_sequence_first100",
    }


def sample_middle_contiguous(seq, out_len):
    start = (seq.shape[0] - out_len) // 2
    return seq[start:start + out_len].copy(), {
        "start": start,
        "mode": "per_sequence_middle100",
    }


def sample_last_contiguous(seq, out_len):
    start = seq.shape[0] - out_len
    return seq[start:start + out_len].copy(), {
        "start": start,
        "mode": "per_sequence_last100",
    }


def sample_random_contiguous(seq, out_len):
    start = random.randint(0, seq.shape[0] - out_len)
    return seq[start:start + out_len].copy(), {
        "start": start,
        "mode": "per_sequence_random100",
    }


# Backward-compatible name: the old contiguous_100 used this random contiguous crop.
def sample_contiguous(seq, out_len):
    return sample_random_contiguous(seq, out_len)


def sample_nonoverlap_piece_starts(seq_len, piece_len, num_pieces):
    intervals = []
    max_start = seq_len - piece_len
    tries = 0
    max_tries = 20000
    while len(intervals) < num_pieces and tries < max_tries:
        tries += 1
        s = random.randint(0, max_start)
        e = s + piece_len
        ok = True
        for old_s, old_e in intervals:
            if not (e <= old_s or s >= old_e):
                ok = False
                break
        if ok:
            intervals.append((s, e))
    if len(intervals) < num_pieces:
        intervals = []
        starts = np.linspace(0, seq_len - piece_len, num_pieces, dtype=np.int64)
        for s in starts:
            intervals.append((int(s), int(s) + piece_len))
    intervals = sorted(intervals, key=lambda x: x[0])
    return [s for s, _ in intervals]


def sample_nonoverlap_pieces_chrono(seq, out_len, num_pieces):
    if out_len % num_pieces != 0:
        raise ValueError("out_len={} must be divisible by num_pieces={}".format(out_len, num_pieces))
    piece_len = out_len // num_pieces
    starts = sample_nonoverlap_piece_starts(seq.shape[0], piece_len, num_pieces)
    pieces = [seq[s:s + piece_len] for s in starts]
    out = np.concatenate(pieces, axis=0).astype(np.float32)
    return out, {"mode": "nonoverlap_pieces_chrono", "num_pieces": num_pieces, "piece_len": piece_len, "starts_head": starts[:10]}


def sample_no_adjacent_points_chrono(seq, out_len):
    n = seq.shape[0]
    if n < 2 * out_len - 1:
        raise ValueError("Sequence length {} too short for {} non-adjacent points".format(n, out_len))
    selected, blocked = [], set()
    tries, max_tries = 0, 100000
    while len(selected) < out_len and tries < max_tries:
        tries += 1
        idx = random.randint(0, n - 1)
        if idx in blocked:
            continue
        selected.append(idx)
        blocked.add(idx)
        if idx - 1 >= 0:
            blocked.add(idx - 1)
        if idx + 1 < n:
            blocked.add(idx + 1)
    if len(selected) < out_len:
        selected = list(range(0, 2 * out_len, 2))
    selected = sorted(selected)
    out = seq[selected].astype(np.float32)
    return out, {"mode": "points_no_adjacent_chrono", "num_points": out_len, "indices_head": selected[:10]}


def build_random_motion_for_method(seq, method, out_len):
    if method == "contiguous_100":
        # Old name kept for compatibility. It is identical to per_sequence_random100.
        return sample_random_contiguous(seq, out_len)
    if method == "per_sequence_first100":
        return sample_first_contiguous(seq, out_len)
    if method == "per_sequence_middle100":
        return sample_middle_contiguous(seq, out_len)
    if method == "per_sequence_last100":
        return sample_last_contiguous(seq, out_len)
    if method == "per_sequence_random100":
        return sample_random_contiguous(seq, out_len)
    if method == "piece_10_nonoverlap_chrono":
        return sample_nonoverlap_pieces_chrono(seq, out_len, num_pieces=10)
    if method == "piece_20_nonoverlap_chrono":
        return sample_nonoverlap_pieces_chrono(seq, out_len, num_pieces=20)
    if method == "piece_50_nonoverlap_chrono":
        return sample_nonoverlap_pieces_chrono(seq, out_len, num_pieces=50)
    if method == "points_100_no_adjacent_chrono":
        return sample_no_adjacent_points_chrono(seq, out_len)
    raise ValueError("Unknown random method: {}".format(method))


def save_random_npz(path, pool, method, synthetic_len, feature_type="expmap", feature_dim=99):
    ensure_dir(os.path.dirname(path))
    subjects, actions, trials, lengths, raw_paths, motions, random_meta = [], [], [], [], [], [], []
    for info in pool.seq_infos:
        key = info["key"]
        random_motion, meta = build_random_motion_for_method(pool.by_key[key], method, synthetic_len)
        subjects.append(info["subject"])
        actions.append(info["action"])
        trials.append(info["trial"])
        lengths.append(random_motion.shape[0])
        raw_paths.append("random://{}/{}".format(method, key))
        motions.append(random_motion)
        random_meta.append({"key": key, "method": method, "original_length": info["original_length"], "meta": meta})

    for entry in pool.get_heldout_entries():
        subjects.append(entry["subject"])
        actions.append(entry["action"])
        trials.append(entry["trial"])
        lengths.append(entry["length"])
        raw_paths.append(entry["raw_path"])
        motions.append(entry["motion"])
        random_meta.append({"heldout": True})

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
        random_method=np.array(method, dtype=object),
        random_synthetic_len=np.array(synthetic_len, dtype=np.int64),
        random_meta=np.array(random_meta, dtype=object),
    )
    return path


# ============================================================
# Dataset / sampler / evaluation
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
    return [round(ret[key], 1) for key in RESULT_KEYS]


def velocity_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)
    pred_vel = pred_xyz66[:, 1:] - pred_xyz66[:, :-1]
    gt_vel = gt_xyz66[:, 1:] - gt_xyz66[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)


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


def train_and_evaluate(train_npz_path, method_name):
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
    train_dataset = H36MExpmapWindowDataset(cfg.train_npz_path, cfg.train_subjects, input_len, output_len, sample_rate=cfg.sample_rate)
    train_batch_sampler = SequenceWindowBatchSampler(train_dataset, batch_size=cfg.batch_size, shuffle_sequences=True, shuffle_windows=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.num_workers, pin_memory=True)
    print("Train NPZ:", cfg.train_npz_path)
    print("Method:", method_name)
    print("Train batches are sequence-pure: one batch contains windows from one random synthetic sequence only.")
    print("Adaptive batch sampler:", "batch_size <=", cfg.batch_size, "num_batches =", len(train_batch_sampler))
    log_dir = resolve_path(cfg.log_dir)
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "train_random_{}.txt".format(method_name))
    acc_log = open(log_path, "w")
    acc_log.write("Seed : {}\n".format(cfg.seed))
    acc_log.write("Method : {}\n".format(method_name))
    acc_log.write("Train NPZ : {}\n".format(cfg.train_npz_path))
    acc_log.write("Test NPZ : {}\n".format(cfg.test_npz_path))
    nb_iter, avg_loss = 0, 0.0
    while nb_iter < cfg.total_iters:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step(model, past_expmap, future_expmap, optimizer, output_len)
            nb_iter += 1
            avg_loss += loss
            if nb_iter % cfg.print_every == 0:
                print("[{}] train iter {} loss={:.6f}".format(method_name, nb_iter, avg_loss / cfg.print_every))
                avg_loss = 0.0
            if nb_iter == cfg.total_iters:
                break
    metrics = evaluate_mpjpe(model, cfg.test_npz_path, cfg.test_subjects, config)
    print("[{}] Final MPJPE {}: {}".format(method_name, RESULT_KEYS, metrics))
    acc_log.write("final\n{}\n".format(" ".join(str(v) for v in metrics)))
    acc_log.flush()
    acc_log.close()
    print("[{}] Saved train/eval log to {}".format(method_name, log_path))
    return metrics


def build_all_random_datasets():
    cfg = RandomBuildConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    pool = OriginalSequencePool(cfg.data_path, include_subjects=DISTILL_SUBJECTS, heldout_subjects=HELDOUT_SUBJECTS, min_len=cfg.synthetic_len)
    output_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(root_dir(), cfg.output_dir)
    ensure_dir(output_dir)
    print("Num training sequences:", len(pool.keys))
    print("Synthetic length:", cfg.synthetic_len)
    print("Output dir:", output_dir)
    print("Old baseline note: contiguous_100 == per_sequence_random100")
    print("Random-position sanity baselines: first100 / middle100 / last100 / random100")
    method_to_path = {}
    for method in cfg.methods:
        save_name = "{}_{}.npz".format(cfg.save_prefix, method)
        save_path = os.path.join(output_dir, save_name)
        save_random_npz(save_path, pool, method=method, synthetic_len=cfg.synthetic_len, feature_dim=cfg.feature_dim)
        method_to_path[method] = save_path
        print("Saved random dataset:", method, "->", save_path)
    return method_to_path


def main():
    method_to_path = build_all_random_datasets()
    summary = {}
    for method, npz_path in method_to_path.items():
        print("\n" + "=" * 80)
        print("Running train/eval for random method:", method)
        print("=" * 80)
        metrics = train_and_evaluate(npz_path, method)
        summary[method] = metrics
    print("\n" + "=" * 80)
    print("Summary")
    print("RESULT_KEYS:", RESULT_KEYS)
    for method, metrics in summary.items():
        print("{}: {}".format(method, metrics))


if __name__ == "__main__":
    main()
