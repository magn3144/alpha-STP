#!/bin/bash
#BSUB -J llm-expert
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -oo jobs/logs/rl_llm_expert_%J.out
#BSUB -eo jobs/logs/rl_llm_expert_%J.err

source "$LS_SUBCWD/jobs/common.sh"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/rl_llm_expert_2x_L40S.yaml \
    --start-round "$start_round"
