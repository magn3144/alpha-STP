#!/bin/bash
#BSUB -J stp-rl-pilot
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/rl_pilot_%J.out
#BSUB -eo jobs/logs/rl_pilot_%J.err

source "$LS_SUBCWD/jobs/common.sh"
: "${LSB_JOBID:?Run this script through bsub}"

pilot_dir="$STORAGE/RL_runtime_pilot/$LSB_JOBID/round0"

python -u RL/RL_step1_generate.py \
    --model "$STORAGE/models/deepseek-coder-1.3b-base" \
    --exp_dir "$pilot_dir" \
    --seed 0 \
    --temperature 1.0 \
    --dataset_config RL/dataset_configs/leanworkbook.json \
    --sampler Sampler_base \
    --conjecture_multiplier 1 \
    --samples_per_statement 32 \
    --dataset_size 256
