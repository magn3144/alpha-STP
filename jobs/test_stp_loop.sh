#!/bin/bash
#BSUB -J stp-loop-test
#BSUB -q gpua100
#BSUB -W 8:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 10GB
#BSUB -oo jobs/logs/test_stp_loop_%J.out
#BSUB -eo jobs/logs/test_stp_loop_%J.err

source "$LS_SUBCWD/jobs/common.sh"
export WANDB_MODE=offline

model="$STORAGE/models/STP_model_Lean_0320"
exp_dir="$STORAGE/STP_tests/loop_$LSB_JOBID"
test -f "$model/config.json"

python -u RL/run_rl_steps.py \
    --exp-dir "$exp_dir" \
    --base-model "$model" \
    --start-round 0 \
    --total-rounds 1 \
    --statements-per-round 20 \
    --samples-per-statement 32
