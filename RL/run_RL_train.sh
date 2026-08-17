#!/bin/bash

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by LSF}"
: "${LSB_DJOB_NUMPROC:?LSB_DJOB_NUMPROC must be set by LSF}"
: "${STORAGE:?Set STORAGE to a local DTU filesystem path}"
: "${EXP_DIR:?Set EXP_DIR to the final training output directory}"
: "${TRAIN_FROM:?Set TRAIN_FROM to a local path or Hugging Face model name}"
: "${SFT_DATASET:?Set SFT_DATASET to the local SFT dataset path}"
: "${MERGE_FROM:?Set MERGE_FROM to the self-play experiment directory}"
: "${MERGE_FROM_ROUNDS:?Set MERGE_FROM_ROUNDS}"

DATASET_CONFIG="./dataset_configs/leanworkbook.json"

python -u RL_step3_final_model.py \
    --base_model "$TRAIN_FROM" \
    --exp_dir "$EXP_DIR" \
    --sft_dataset "$SFT_DATASET" \
    --dataset_config "$DATASET_CONFIG" \
    --epoch 1 \
    --lr 1e-4 \
    --include_synthetic_examples \
    --merge_from "$MERGE_FROM" \
    --merge_from_rounds "$MERGE_FROM_ROUNDS"
