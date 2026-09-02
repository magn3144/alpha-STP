#!/bin/bash
#BSUB -J stp-sft-p16-v100
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/sft_pass16_v100_%J.out
#BSUB -eo jobs/logs/sft_pass16_v100_%J.err

source "$LS_SUBCWD/jobs/common.sh"

PYTHONPATH=RL python -u scripts/eval_sft_pass_at_k.py \
    --model "$STORAGE/SFT/moy6hpa5/step-1004" \
    --validation-data "$STORAGE/data/SFT/validation.json" \
    --output-dir "$STORAGE/evals/sft_moy6hpa5_step1004_validation100_pass16_seed42_fixed" \
    --prompt-tokens 1536 \
    --max-new-tokens 512
