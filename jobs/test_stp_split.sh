#!/bin/bash
#BSUB -J stp-split-test
#BSUB -q gpua100
#BSUB -W 8:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 64GB
#BSUB -oo jobs/logs/test_stp_split_%J.out
#BSUB -eo jobs/logs/test_stp_split_%J.err

source "$LS_SUBCWD/jobs/common.sh"
export WANDB_MODE=offline

model="$STORAGE/models/STP_model_Lean_0320"
round_dir="$STORAGE/STP_tests/split_$LSB_JOBID/round0"
test -f "$model/config.json"

python -u RL/RL_step1_generate.py \
    --model "$model" \
    --exp_dir "$round_dir" \
    --seed 0 \
    --temperature 1.0 \
    --dataset_config RL/dataset_configs/leanworkbook.json \
    --sampler Sampler_base \
    --conjecture_multiplier 1 \
    --samples_per_statement 4 \
    --statements_per_round 3

ray stop --force
nvidia-smi

python -u RL/RL_step2_train.py \
    --base_model "$model" \
    --exp_dir "$round_dir" \
    --epoch 1 \
    --lr 5e-5
