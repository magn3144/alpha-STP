#!/bin/bash
#BSUB -J stp-sft-l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 24GB
#BSUB -oo jobs/logs/sft_l40s_%J.out
#BSUB -eo jobs/logs/sft_l40s_%J.err

source "$LS_SUBCWD/jobs/common.sh"

python -u RL/run_sft.py \
    --storage "$STORAGE" \
    --config jobs/yaml/sft_L40S.yaml
