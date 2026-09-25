#!/bin/bash
#BSUB -J sft-deltaproof-a100
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_deltaproof_a100_%J.out
#BSUB -eo jobs/logs/sft_deltaproof_a100_%J.err

source "$LS_SUBCWD/jobs/bash/common.sh"

python -u RL/run_sft.py \
    --config jobs/yaml/sft/sft_deltaproof_A100_40GB.yaml
