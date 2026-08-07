#!/bin/sh
#BSUB -q gpuv100
#BSUB -J alphaproof_retries
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /work3/s204164/alpha-STP/data/evaluations/alphaproof_retries_%J.out
#BSUB -e /work3/s204164/alpha-STP/data/evaluations/alphaproof_retries_%J.err

set -eu

cd /work3/s204164/alpha-STP
module load cuda/12.6.3
. .venv/bin/activate

export OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"
export MKL_NUM_THREADS="$LSB_DJOB_NUMPROC"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -u -m stp.evaluate_difficulty_score.evaluate_alphaproof_retries \
    --config configs/alpha_stp.toml \
    --name alphaproof-retries
