#!/bin/bash
#BSUB -J stp-eval
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/generation_and_test_%J.out
#BSUB -eo jobs/logs/generation_and_test_%J.err

source "$LS_SUBCWD/jobs/common.sh"
: "${EVALUATION_MODEL:?Set EVALUATION_MODEL in jobs/config.sh}"

python -u RL/run_generation_and_test.py \
    --model "$EVALUATION_MODEL" \
    --exp-dir "$STORAGE/STP/benchmark_results"
