#!/bin/bash
#BSUB -J deltaproof-baseline
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -oo jobs/logs/rl_deltaproof_baseline_%J.out
#BSUB -eo jobs/logs/rl_deltaproof_baseline_%J.err

source "$LS_SUBCWD/jobs/common.sh"
module load cuda/12.6.3
export DELTA_PROOF="${DELTA_PROOF:-$LS_SUBCWD/../delta-proof}"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/rl_deltaproof_baseline_2x_L40S.yaml \
    --start-round "$start_round"
