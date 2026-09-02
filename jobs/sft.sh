#!/bin/bash
#BSUB -J stp-sft
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -oo jobs/logs/sft_%J.out
#BSUB -eo jobs/logs/sft_%J.err

source "$LS_SUBCWD/jobs/common.sh"

python -u RL/run_sft.py \
    --config jobs/yaml/sft_L40S.yaml
