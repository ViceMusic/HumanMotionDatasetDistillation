import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backbones.simlpe import SiMLPeMotionBackbone, build_simlpe_config, expmap_to_simlpe_xyz66, expmap_to_xyz32


# TRAIN_NPZ_PATH = "datasets/h36m_expmap_sequences_distilled.npz"
# TEST_NPZ_PATH = "datasets/h36m_expmap_sequences_distilled.npz"

TRAIN_NPZ_PATH="datasets/h36m_expmap_sequences_distilled_bs=64_iter=8000_1backbone_no_reinit_traj.npz"
TEST_NPZ_PATH="/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"

LOG_PATH = "logs/train_acc__bs64_iter8000.txt"

TRAIN_SUBJECTS = ["S1", "S6", "S7", "S8", "S9", "S11"]
TEST_SUBJECTS = ["S5"]

sample_rate=1
BATCH_SIZE = 64
NUM_WORKERS = 4
TOTAL_ITERS = 8000
PRINT_EVERY = 100
SEED = 888
LR = 3e-4
WEIGHT_DECAY = 1e-4
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


def velocity_loss(pred_xyz66, gt_expmap):
    gt_xyz66 = expmap_to_simlpe_xyz66(gt_expmap)
    pred_vel = pred_xyz66[:, 1:] - pred_xyz66[:, :-1]
    gt_vel = gt_xyz66[:, 1:] - gt_xyz66[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)


class H36MExpmapWindowDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=sample_rate):
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


class H36MExpmapEvalDataset(Dataset):
    def __init__(self, npz_path, subjects, input_len, output_len, shift_step=1, sample_rate=sample_rate):
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = build_simlpe_config()
    input_len = config.motion.h36m_input_length
    output_len = config.motion.h36m_target_length_train

    model = SiMLPeMotionBackbone(config).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_dataset = H36MExpmapWindowDataset(TRAIN_NPZ_PATH, TRAIN_SUBJECTS, input_len, output_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )

    log_path = resolve_path(LOG_PATH)
    ensure_dir(os.path.dirname(log_path))
    acc_log = open(log_path, "w")
    acc_log.write("Seed : {}\n".format(SEED))

    nb_iter = 0
    avg_loss = 0.0
    while nb_iter < TOTAL_ITERS:
        for past_expmap, future_expmap in train_loader:
            loss, loss_pred, loss_vel = train_step(model, past_expmap, future_expmap, optimizer, output_len)
            nb_iter += 1
            avg_loss += loss

            if nb_iter % PRINT_EVERY == 0:
                print("Iter {} loss={:.6f}".format(nb_iter, avg_loss / PRINT_EVERY))
                avg_loss = 0.0

            if nb_iter == TOTAL_ITERS:
                break

    metrics = evaluate_mpjpe(model, TEST_NPZ_PATH, TEST_SUBJECTS, config)
    print("Final MPJPE {}: {}".format(RESULT_KEYS, metrics))
    acc_log.write("final\n{}\n".format(" ".join(str(v) for v in metrics)))
    acc_log.flush()
    acc_log.close()


if __name__ == "__main__":
    main()
