#!/bin/bash
#BSUB -J stp-cache-sft
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo jobs/logs/cache_sft_%J.out
#BSUB -eo jobs/logs/cache_sft_%J.err

set -euo pipefail

: "${LS_SUBCWD:?Submit jobs from the repository root with bsub < jobs/cache_sft.sh}"
cd "$LS_SUBCWD"

module load python3/3.10.18
source "$LS_SUBCWD/.venv/bin/activate"

export STORAGE="$LS_SUBCWD/storage"
export TOKENIZERS_PARALLELISM=false

python --version
python -u RL/run_sft.py \
    --config jobs/yaml/sft_L40S.yaml \
    --cache-only
