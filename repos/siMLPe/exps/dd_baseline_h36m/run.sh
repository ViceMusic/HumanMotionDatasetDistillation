#!/usr/bin/env bash

python train.py \
  --exp-name log/dd_h36m_acc.txt \
  --data-path /home/user/workspace/HumanMotionDatasetDistillation/datasets/processed/Human3.6m/h36m_expmap_sequences.npz \
  --with-normalization \
  --num 48
