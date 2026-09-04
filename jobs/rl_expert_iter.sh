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

start_round="${START_ROUND:-0}"

python -u RL/run_rl_expert_iter.py \
    --config jobs/yaml/rl_expert_iter_A100.yaml \
    --start-round "$start_round"
