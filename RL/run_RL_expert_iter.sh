#!/bin/bash

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by LSF}"
: "${LSB_DJOB_NUMPROC:?LSB_DJOB_NUMPROC must be set by LSF}"
: "${STORAGE:?Set STORAGE to a local DTU filesystem path}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output directory}"
: "${BASE_MODEL:?Set BASE_MODEL to a local path or Hugging Face model name}"
: "${DATASET_CONFIG:?Set DATASET_CONFIG to the dataset configuration path}"
: "${START_ROUND:?Set START_ROUND}"
: "${TOTAL_ROUNDS:?Set TOTAL_ROUNDS}"

for ((ROUND=START_ROUND; ROUND<TOTAL_ROUNDS; ROUND++)); do
    PREV_ROUND=$((ROUND-1))
    if [ "$ROUND" -eq 0 ]; then
        MODEL="$BASE_MODEL"
    else
        MODEL="$EXP_DIR/round${PREV_ROUND}/RL_model"
    fi

    CURRENT_EXP_DIR="$EXP_DIR/round${ROUND}"
    echo "Starting expert iteration round ${ROUND} with model ${MODEL}"

    python -u RL_step1_generate.py \
        --model "$MODEL" \
        --exp_dir "$CURRENT_EXP_DIR" \
        --seed "$ROUND" \
        --temperature 1.0 \
        --dataset_config "$DATASET_CONFIG" \
        --sampler "Sampler_naive" \
        --samples_per_statement 64 \
        --statements_per_round 0

    python -u RL_step3_final_model.py \
        --base_model "$BASE_MODEL" \
        --exp_dir "$CURRENT_EXP_DIR" \
        --dataset_config "$DATASET_CONFIG" \
        --epoch 1 \
        --lr 5e-5 \
        --merge_from "$EXP_DIR" \
        --merge_from_rounds "$((ROUND+1))"
done

echo "All expert iteration rounds completed successfully."
