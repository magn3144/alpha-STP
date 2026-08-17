#!/bin/bash
#BSUB -J stp-expert
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/rl_expert_iter_%J.out
#BSUB -eo jobs/logs/rl_expert_iter_%J.err

source "$LS_SUBCWD/jobs/common.sh"
: "${BASE_MODEL:?Set BASE_MODEL in jobs/config.sh}"

python -u RL/run_rl_expert_iter.py \
    --exp-dir "$STORAGE/Expit_LeanWorkbook" \
    --base-model "$BASE_MODEL" \
    --start-round 0 \
    --total-rounds 12
