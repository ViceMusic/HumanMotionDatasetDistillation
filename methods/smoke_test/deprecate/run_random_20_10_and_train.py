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
# Random 20->10 baseline for H36M motion prediction
# Save as:
#   methods/smoke_test/run_random_20_10_and_train.py
# Run:
#   python methods/smoke_test/run_random_20_10_and_train.py
#
# Rule:
#   For each training sequence, randomly crop one contiguous 30-frame segment.
#   20 frames are used as input and 10 frames as prediction target.
#   Train on S1/S6/S7/S8/S9/S11; evaluate on original full S5.
# ============================================================


DISTILL_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
TEST_SUBJECTS = ["S5"]

INPUT_LEN_20_10 = 20
OUTPUT_LEN_20_10 = 10
WINDOW_LEN_20_10 = INPUT_LEN_20_10 + OUTPUT_LEN_20_10

RESULT_KEYS = ["#2", "#4", "#8", "#10"]
JOINT_USED_XYZ = np.array(
    [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
).astype(np.int64)
JOINT_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
JOINT_EQUAL = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)


def configure_motion_lengths(config):
    config.motion.h36m_input_length = INPUT_LEN_20_10
    config.motion.h36m_target_length_train = OUTPUT_LEN_20_10
    config.motion.h36m_target_length_eval = OUTPUT_LEN_20_10
    return config


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


@dataclass
class RandomConfig:
    data_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    output_dir: str = "datasets"
    save_name: str = "h36m_expmap_sequences_random_20_10_len30_seed888.npz"
    seed: int = 888
    crop_len: int = WINDOW_LEN_20_10
    feature_dim: int = 99


@dataclass
class TrainConfig:
    train_npz_path: str = ""
    test_npz_path: str = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
    log_path: str = "logs/train_random_20_10_len30_seed888_iter8000.txt"

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


def build_random_crop_npz():
    cfg = RandomConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data = np.load(resolve_path(cfg.data_path), allow_pickle=True)
    wanted = set(DISTILL_SUBJECTS)

    subjects = []
    actions = []
    trials = []
    lengths = []
    raw_paths = []
    motions = []
    crop_logs = []

    zipped = zip(
        data["subjects"],
        data["actions"],
        data["trials"],
        data["raw_paths"],
        data["motions"],
    )

    for idx, (subject, action, trial, raw_path, motion) in enumerate(zipped):
        subject_name = normalize_subject(subject)
        if subject_name not in wanted:
            continue

        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != cfg.feature_dim:
            continue
        if motion.shape[0] < cfg.crop_len:
            continue

        start = random.randint(0, motion.shape[0] - cfg.crop_len)
        crop = motion[start : start + cfg.crop_len].astype(np.float32)

        subjects.append(subject_name)
        actions.append(as_text(action))
        trials.append(as_text(trial))
        lengths.append(cfg.crop_len)
        raw_paths.append("random20_10://idx_{}_start_{}__{}".format(idx, start, as_text(raw_path)))
        motions.append(crop)
        crop_logs.append(
            {
                "idx": idx,
                "subject": subject_name,
                "action": as_text(action),
                "trial": as_text(trial),
                "original_length": int(motion.shape[0]),
                "start": int(start),
                "crop_len": int(cfg.crop_len),
            }
        )

    if not motions:
        raise ValueError("No valid random crops found from {}".format(cfg.data_path))

    motion_array = np.empty(len(motions), dtype=object)
    for i, motion in enumerate(motions):
        motion_array[i] = motion

    output_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(root_dir(), cfg.output_dir)
    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, cfg.save_name)

    np.savez(
        save_path,
        subjects=np.array(subjects, dtype=object),
        actions=np.array(actions, dtype=object),
        trials=np.array(trials, dtype=object),
        lengths=np.array(lengths, dtype=np.int64),
        raw_paths=np.array(raw_paths, dtype=object),
        motions=motion_array,
        feature_type=np.array("expmap", dtype=object),
        feature_dim=np.array(cfg.feature_dim, dtype=np.int64),
        random_seed=np.array(cfg.seed, dtype=np.int64),
        crop_len=np.array(cfg.crop_len, dtype=np.int64),
        input_len=np.array(INPUT_LEN_20_10, dtype=np.int64),
        output_len=np.array(OUTPUT_LEN_20_10, dtype=np.int64),
        crop_logs=np.array(crop_logs, dtype=object),
    )

    print("Saved random 20->10 crop dataset to {}".format(save_path))
    print("Num random crops:", len(motions))
    print("Each crop length:", cfg.crop_len)
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

        print("Eval subjects:", subjects)
        print("Num eval sequences:", len(self.seqs))
        print("Num eval windows:", len(self.data_idx))
        print("First 10 eval starts:", self.data_idx[:10])
        print("Last 10 eval starts:", self.data_idx[-10:])

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        seq_idx, start = self.data_idx[index]
        window = self.seqs[seq_idx][start : start + self.total_len]
        motion_xyz32 = expmap_to_xyz32(window.unsqueeze(0)).squeeze(0) / 1000.0
        return window[: self.input_len], motion_xyz32[self.input_len :]


def velocity_loss(pred_xyz66, gt_future_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_future_expmap)
    pred_vel = pred_xyz66[:, 1:] - pred_xyz66[:, :-1]
    gt_vel = gt_xyz66[:, 1:] - gt_xyz66[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)


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

    config = configure_motion_lengths(build_simlpe_config())
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

    print("20->10 mode: input_len=20, output_len=10, window_len=30")
    print("Train batches are sequence-pure: one batch contains windows from one random crop only.")
    print("Train windows:", len(train_dataset))
    print("Adaptive batch sampler:", "batch_size <=", cfg.batch_size, "num_batches =", len(train_batch_sampler))

    log_path = resolve_path(cfg.log_path)
    ensure_dir(os.path.dirname(log_path))
    acc_log = open(log_path, "w")
    acc_log.write("Seed : {}\n".format(cfg.seed))
    acc_log.write("Train NPZ : {}\n".format(cfg.train_npz_path))
    acc_log.write("Test NPZ : {}\n".format(cfg.test_npz_path))
    acc_log.write("Input/Output : 20/10\n")

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
    random_path = build_random_crop_npz()
    train_and_evaluate(random_path)


if __name__ == "__main__":
    main()
