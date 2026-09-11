#!/bin/bash

set -euo pipefail

: "${LS_SUBCWD:?Submit jobs from the repository root with bsub < jobs/JOB.sh}"
cd "$LS_SUBCWD"

module load python3/3.10.18
source "$LS_SUBCWD/.venv/bin/activate"

export DATA="$LS_SUBCWD/data"
export TOKENIZERS_PARALLELISM=false

python --version
nvidia-smi -L
