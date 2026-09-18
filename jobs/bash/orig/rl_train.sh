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

source "$LS_SUBCWD/jobs/bash/common.sh"

base_model=deepseek-ai/DeepSeek-Prover-V1.5-SFT

python -u RL/run_rl_train.py \
    --exp-dir "$DATA/runs/final_training_deepseek_prover_v1.5_a100_leanworkbook" \
    --train-from "$base_model" \
    --sft-dataset "$DATA/dataset/prover_sft/mathlib.json" \
    --merge-from "$DATA/runs/stp_deepseek_prover_v1.5_a100_leanworkbook" \
    --merge-from-rounds 12 \
    --training-config jobs/yaml/rl_2x_A100_40gb.yaml
