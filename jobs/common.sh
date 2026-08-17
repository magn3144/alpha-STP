#!/bin/bash

set -euo pipefail

: "${LS_SUBCWD:?Submit jobs from the repository root with bsub < jobs/JOB.sh}"
cd "$LS_SUBCWD"

if [[ ! -f jobs/config.sh ]]; then
    echo "Copy jobs/config.sh.example to jobs/config.sh and fill in the DTU paths." >&2
    exit 1
fi

source jobs/config.sh

: "${STP_VENV:?Set STP_VENV in jobs/config.sh}"
: "${STP_MODULES:?Set STP_MODULES in jobs/config.sh}"
: "${STORAGE:?Set STORAGE in jobs/config.sh}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in jobs/config.sh}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in jobs/config.sh}"

read -r -a modules <<< "$STP_MODULES"
module load "${modules[@]}"
source "$STP_VENV/bin/activate"

export STORAGE WANDB_ENTITY WANDB_PROJECT
export TOKENIZERS_PARALLELISM=false

python --version
nvidia-smi -L
