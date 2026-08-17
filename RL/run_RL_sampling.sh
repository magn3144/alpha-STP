#!/bin/bash

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by LSF}"
: "${LSB_DJOB_NUMPROC:?LSB_DJOB_NUMPROC must be set by LSF}"
: "${STORAGE:?Set STORAGE to a local DTU filesystem path}"
: "${EXP_DIR:?Set EXP_DIR to the sampling output directory}"
: "${BASE_MODEL:?Set BASE_MODEL to a local path or Hugging Face model name}"
: "${DATASET_CONFIG:?Set DATASET_CONFIG to the dataset configuration path}"
: "${START_ROUND:?Set START_ROUND}"
: "${TOTAL_ROUNDS:?Set TOTAL_ROUNDS}"

for ((ROUND=START_ROUND; ROUND<TOTAL_ROUNDS; ROUND++)); do
    MODEL="$BASE_MODEL"
    SEED="$ROUND"
    SPL=64
    CURRENT_EXP_DIR="$EXP_DIR/round${ROUND}"

    echo "Starting sampling round ${ROUND} with model ${MODEL}"
    python -u RL_step1_generate.py \
        --model "$MODEL" \
        --exp_dir "$CURRENT_EXP_DIR" \
        --seed "$SEED" \
        --temperature 1.0 \
        --dataset_config "$DATASET_CONFIG" \
        --sampler "Sampler_naive" \
        --samples_per_statement "$SPL" \
        --statements_per_round 0
done

echo "All sampling rounds completed successfully."
