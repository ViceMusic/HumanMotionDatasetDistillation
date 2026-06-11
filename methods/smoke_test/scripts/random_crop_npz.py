import os
import random

import numpy as np


# Manually edit these values before running on the server.
INPUT_NPZ_PATH = "/home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz"
OUTPUT_NPZ_NAME = "h36m_expmap_sequences_random100.npz"
CROP_LEN = 100
SEED = 888
KEEP_SHORT = False


def as_text(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    elif value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def smoke_test_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def output_path():
    return os.path.join(smoke_test_root(), "datasets", OUTPUT_NPZ_NAME)


def random_crop_motion(motion, crop_len):
    if motion.shape[0] < crop_len:
        if KEEP_SHORT:
            return motion, 0
        return None, None
    max_start = motion.shape[0] - crop_len
    start = random.randint(0, max_start)
    return motion[start : start + crop_len], start


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    data = np.load(INPUT_NPZ_PATH, allow_pickle=True)
    subjects = []
    actions = []
    trials = []
    lengths = []
    raw_paths = []
    motions = []

    for idx, motion in enumerate(data["motions"]):
        motion = np.asarray(motion, dtype=np.float32)
        cropped, start = random_crop_motion(motion, CROP_LEN)
        if cropped is None:
            continue

        subjects.append(data["subjects"][idx])
        actions.append(data["actions"][idx])
        trials.append(data["trials"][idx])
        lengths.append(cropped.shape[0])
        raw_path = as_text(data["raw_paths"][idx])
        raw_paths.append("{}#random_crop_start={};len={}".format(raw_path, start, cropped.shape[0]))
        motions.append(cropped)

    save_path = output_path()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    motion_array = np.empty(len(motions), dtype=object)
    for idx, motion in enumerate(motions):
        motion_array[idx] = motion

    np.savez(
        save_path,
        subjects=np.array(subjects, dtype=object),
        actions=np.array(actions, dtype=object),
        trials=np.array(trials, dtype=object),
        lengths=np.array(lengths, dtype=np.int64),
        raw_paths=np.array(raw_paths, dtype=object),
        motions=motion_array,
        feature_type=data["feature_type"],
        feature_dim=data["feature_dim"],
    )
    print("Saved {} cropped sequences to {}".format(len(motions), save_path))


if __name__ == "__main__":
    main()
