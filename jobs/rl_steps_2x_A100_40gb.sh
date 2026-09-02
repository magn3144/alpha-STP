#!/bin/bash
#BSUB -J stp-rl-small
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/rl_steps_small_%J.out
#BSUB -eo jobs/logs/rl_steps_small_%J.err

source "$LS_SUBCWD/jobs/common.sh"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/rl_2x_A100_40gb.yaml \
    --start-round "$start_round"
