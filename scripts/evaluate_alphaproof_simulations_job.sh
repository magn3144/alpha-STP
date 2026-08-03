#!/bin/sh
#BSUB -q gpuv100
#BSUB -J alphaproof_simulations
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -o /work3/s204164/alpha-STP/data/evaluations/alphaproof_simulations_%J.out
#BSUB -e /work3/s204164/alpha-STP/data/evaluations/alphaproof_simulations_%J.err

set -eu

cd /work3/s204164/alpha-STP
module load cuda/12.6.3
. .venv/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"
export MKL_NUM_THREADS="$LSB_DJOB_NUMPROC"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -u scripts/evaluate_alphaproof_simulations.py \
    --config configs/alpha_stp.toml \
    --problem-index 0 \
    --name "alphaproof-simulations-${LSB_JOBID}"
