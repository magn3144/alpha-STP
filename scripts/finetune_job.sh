#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finetune_codegen2_1b
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /work3/s204164/alpha-STP/data/training_runs/codegen2-1b-sft/finetune_%J.out
#BSUB -e /work3/s204164/alpha-STP/data/training_runs/codegen2-1b-sft/finetune_%J.err

set -eu

cd /work3/s204164/alpha-STP
module load cuda/12.6.3
. .venv/bin/activate

set -a
. ./.env
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in .env}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -u -m stp.finetuning.finetune codegen2-1b-sft \
    --epochs 1 \
    --train-microbatch-size 1 \
    --validation-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --gradient-checkpointing \
    --wandb-mode online \
    "$@"
