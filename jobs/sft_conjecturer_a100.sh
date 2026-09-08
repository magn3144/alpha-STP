#!/bin/bash
#BSUB -J stp-sft-conjecturer
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_conjecturer_a100_%J.out
#BSUB -eo jobs/logs/sft_conjecturer_a100_%J.err

source "$LS_SUBCWD/jobs/common.sh"

python -u RL/run_sft.py \
    --config jobs/yaml/sft_conjecturer_A100_80GB.yaml \
    --no-eval
