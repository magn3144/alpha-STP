#!/bin/bash
#BSUB -J stp-sft-deltaproof-1-8-l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_stp_llm_deltaproof_1_8_l40s_%J.out
#BSUB -eo jobs/logs/sft_stp_llm_deltaproof_1_8_l40s_%J.err

source "$LS_SUBCWD/jobs/common.sh"

python -u RL/run_sft.py \
    --config jobs/yaml/sft_stp_llm_deltaproof_1_8_L40S.yaml
