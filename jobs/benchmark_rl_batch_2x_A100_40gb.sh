#!/bin/bash
#BSUB -J stp-rl-batch
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -oo jobs/logs/rl_batch_%J.out
#BSUB -eo jobs/logs/rl_batch_%J.err

source "$LS_SUBCWD/jobs/common.sh"
: "${LSB_JOBID:?Run this script through bsub}"

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u scripts/benchmark_rl_batch.py \
    --model "$STORAGE/models/deepseek-coder-1.3b-base" \
    --config jobs/yaml/rl_batch_benchmark_2x_A100_40gb.yaml \
    --train-data "$STORAGE/data/SFT/train.json" \
    --train-cache "$STORAGE/data/SFT/cache/deepseek-coder-1.3b-base_2048/train" \
    --output-dir "$STORAGE/RL_batch_benchmark/$LSB_JOBID" \
    --upper-global-batch 64
