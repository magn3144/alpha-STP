#!/bin/bash
#BSUB -J stp-final-train
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -oo jobs/logs/rl_train_%J.out
#BSUB -eo jobs/logs/rl_train_%J.err

source "$LS_SUBCWD/jobs/common.sh"

base_model=deepseek-ai/DeepSeek-Prover-V1.5-SFT

python -u RL/run_rl_train.py \
    --exp-dir "$STORAGE/STP_LeanWorkbook_merged" \
    --train-from "$base_model" \
    --sft-dataset "$STORAGE/data/SFT/mathlib.json" \
    --merge-from "$STORAGE/STP_LeanWorkbook" \
    --merge-from-rounds 12 \
    --training-config jobs/yaml/rl_2x_A100_40gb.yaml
