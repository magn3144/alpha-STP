#!/bin/bash
#BSUB -J deltaproof-stp
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -oo jobs/logs/rl_deltaproof_stp_%J.out
#BSUB -eo jobs/logs/rl_deltaproof_stp_%J.err

source "$LS_SUBCWD/jobs/bash/common.sh"
module load cuda/12.6.3
export DELTA_PROOF="${DELTA_PROOF:-$LS_SUBCWD/../delta-proof}"

start_round="${START_ROUND:-0}"

python -u RL/run_rl_steps.py \
    --config jobs/yaml/A100/rl_deltaproof_stp_A100.yaml \
    --start-round "$start_round"
