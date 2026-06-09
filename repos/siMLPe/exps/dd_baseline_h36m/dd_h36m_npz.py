import numpy as np
import torch
import torch.utils.data as data

from utils.misc import expmap2rotmat_torch, find_indices_256, rotmat2xyz_torch


class DDH36MNPZBase(data.Dataset):
    used_joint_indexes = np.array(
        [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
    ).astype(np.int64)

    def __init__(self, config, split_name):
        super(DDH36MNPZBase, self).__init__()
        self.config = config
        self.split_name = split_name
        self.npz_path = config.dd_h36m_npz_path
        self.h36m_motion_input_length = config.motion.h36m_input_length
        self.h36m_motion_target_length = config.motion.h36m_target_length
        self.shift_step = config.shift_step

    @staticmethod
    def _as_scalar(value):
        value = np.asarray(value)
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value

    @classmethod
    def _subject_key(cls, value):
        value = str(cls._as_scalar(value)).strip()
        if value.startswith("S"):
            value = value[1:]
        return value

    def _load_selected_sequences(self, want_test):
        data_npz = np.load(self.npz_path, allow_pickle=True)
        subjects = data_npz["subjects"]
        actions = data_npz["actions"] if "actions" in data_npz.files else [None] * len(subjects)
        trials = data_npz["trials"] if "trials" in data_npz.files else [None] * len(subjects)
        motions = data_npz["motions"]

        sequences = []
        for subject, action, trial, motion in zip(subjects, actions, trials, motions):
            is_s5 = self._subject_key(subject) == "5"
            if is_s5 != want_test:
                continue
            xyz = self._expmap_to_xyz32(motion)
            sequences.append(
                {
                    "subject": self._as_scalar(subject),
                    "action": str(self._as_scalar(action)).lower() if action is not None else None,
                    "trial": self._as_scalar(trial),
                    "xyz": xyz,
                }
            )
        return sequences

    @staticmethod
    def _expmap_to_xyz32(motion):
        pose_info = np.asarray(motion, dtype=np.float32)
        if pose_info.ndim != 2 or pose_info.shape[1] != 99:
            raise ValueError("Expected each motion sequence to have shape [T, 99], got {}".format(pose_info.shape))

        t = pose_info.shape[0]
        pose_info = pose_info.reshape(t, 33, 3)
        pose_info[:, :2] = 0
        pose_info = pose_info.reshape(-1, 3)
        pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(t, 33, 3, 3)[:, 1:]
        xyz_info = rotmat2xyz_torch(pose_info)
        return xyz_info

    def __len__(self):
        return len(self.data_idx)


class DDH36MNPZDataset(DDH36MNPZBase):
    def __init__(self, config, split_name, data_aug=False):
        super(DDH36MNPZDataset, self).__init__(config, split_name)
        self.data_aug = data_aug
        self.h36m_seqs = []
        self.data_idx = []
        self._collect_all()

    def _collect_all(self):
        sequences = self._load_selected_sequences(want_test=False)
        idx = 0
        for item in sequences:
            h36m_motion_poses = item["xyz"]
            n = len(h36m_motion_poses)
            if n < self.h36m_motion_input_length + self.h36m_motion_target_length:
                continue

            sampled_index = np.arange(0, n, 2)
            h36m_motion_poses = h36m_motion_poses[sampled_index]
            t = h36m_motion_poses.shape[0]
            h36m_motion_poses = h36m_motion_poses[:, self.used_joint_indexes, :].reshape(t, -1)

            valid_frames = np.arange(
                0,
                t - self.h36m_motion_input_length - self.h36m_motion_target_length + 1,
                self.shift_step,
            )
            if len(valid_frames) == 0:
                continue
            self.h36m_seqs.append(h36m_motion_poses)
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        motion = self.h36m_seqs[idx][frame_indexes]
        if self.data_aug and torch.rand(1)[0] > 0.5:
            motion = motion[torch.arange(motion.size(0) - 1, -1, -1)]

        h36m_motion_input = motion[: self.h36m_motion_input_length] / 1000.0
        h36m_motion_target = motion[self.h36m_motion_input_length :] / 1000.0
        return h36m_motion_input.float(), h36m_motion_target.float()


class DDH36MNPZEval(DDH36MNPZBase):
    def __init__(self, config, split_name):
        super(DDH36MNPZEval, self).__init__(config, split_name)
        self.h36m_seqs = []
        self.data_idx = []
        self._collect_all()

    def _collect_all(self):
        sequences = self._load_selected_sequences(want_test=True)
        grouped = {}
        for item in sequences:
            key = item["action"] if item["action"] is not None else "unknown"
            grouped.setdefault(key, []).append(item)

        idx = 0
        for _, items in grouped.items():
            items = sorted(items, key=lambda x: str(x["trial"]))
            if len(items) >= 2:
                idx = self._add_eval_pair(items[0]["xyz"], items[1]["xyz"], idx)
            else:
                idx = self._add_sliding_windows(items[0]["xyz"], idx)

    def _downsample(self, xyz):
        sampled_index = np.arange(0, xyz.shape[0], 2)
        return xyz[sampled_index].reshape(-1, 32, 3)

    def _add_eval_pair(self, xyz0, xyz1, idx):
        poses0 = self._downsample(xyz0)
        poses1 = self._downsample(xyz1)
        total_len = self.h36m_motion_input_length + self.h36m_motion_target_length
        fs_sel1, fs_sel2 = find_indices_256(poses0.shape[0], poses1.shape[0], total_len, input_n=self.h36m_motion_input_length)
        self.h36m_seqs.append(poses0)
        self.h36m_seqs.append(poses1)
        self.data_idx.extend(zip([idx] * len(fs_sel1), fs_sel1[:, 0].tolist()))
        self.data_idx.extend(zip([idx + 1] * len(fs_sel2), fs_sel2[:, 0].tolist()))
        return idx + 2

    def _add_sliding_windows(self, xyz, idx):
        poses = self._downsample(xyz)
        total_len = self.h36m_motion_input_length + self.h36m_motion_target_length
        valid_frames = np.arange(0, poses.shape[0] - total_len + 1, self.shift_step)
        if len(valid_frames) == 0:
            return idx
        self.h36m_seqs.append(poses)
        self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
        return idx + 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        motion = self.h36m_seqs[idx][frame_indexes]

        h36m_motion_input = motion[: self.h36m_motion_input_length] / 1000.0
        h36m_motion_target = motion[self.h36m_motion_input_length :] / 1000.0
        return h36m_motion_input.float(), h36m_motion_target.float()
