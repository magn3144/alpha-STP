#!/bin/sh
#BSUB -q gpuv100
#BSUB -J evaluate_numina
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /work3/s204164/alpha-STP/data/evaluations/evaluate_numina_%J.out
#BSUB -e /work3/s204164/alpha-STP/data/evaluations/evaluate_numina_%J.err

set -eu

cd /work3/s204164/alpha-STP
module load cuda/12.6.3
. .venv/bin/activate

export OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"
export MKL_NUM_THREADS="$LSB_DJOB_NUMPROC"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
stp declarations --config configs/alpha_stp.toml
python -u scripts/evaluate_numina.py \
    --config configs/alpha_stp.toml \
    --llm-model models/Kimina-Prover-Preview-Distill-1.5B \
    --llm-prover-handler kimina_numina \
    --name "numina-${LSB_JOBID}"
