#!/bin/bash
#BSUB -J deltaproof-sampling
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 48
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -oo jobs/logs/rl_deltaproof_sampling_%J.out
#BSUB -eo jobs/logs/rl_deltaproof_sampling_%J.err

source "$LS_SUBCWD/jobs/bash/common.sh"
module load cuda/12.6.3
export DELTA_PROOF="${DELTA_PROOF:-$LS_SUBCWD/../delta-proof}"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/L40S/rl_deltaproof_sampling_L40S.yaml \
    --start-round "$start_round"
