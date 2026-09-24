#!/bin/bash
#BSUB -J sft-deltaproof-l40s
#BSUB -q gpul40s
#BSUB -W 4:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_deltaproof_l40s_%J.out
#BSUB -eo jobs/logs/sft_deltaproof_l40s_%J.err

source "$LS_SUBCWD/jobs/bash/common.sh"

python -u RL/run_sft.py \
    --config jobs/yaml/sft/sft_deltaproof_L40S.yaml
