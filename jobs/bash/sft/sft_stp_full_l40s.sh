#!/bin/bash
#BSUB -J stp-sft-full-l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_stp_full_l40s_%J.out
#BSUB -eo jobs/logs/sft_stp_full_l40s_%J.err

source "$LS_SUBCWD/jobs/bash/common.sh"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python -u RL/run_sft.py \
    --config jobs/yaml/sft/sft_stp_full_L40S.yaml \
    --run-id stpfull20261003
