#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL> <EXP_DIR>"
    exit 1
fi

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by LSF}"
: "${LSB_DJOB_NUMPROC:?LSB_DJOB_NUMPROC must be set by LSF}"
: "${STORAGE:?Set STORAGE to a local DTU filesystem path}"

MODEL=$1
EXP_DIR=$2
DATASET_CONFIG="./dataset_configs/miniF2F_ProofNet.json"

python -u generate_and_test.py --model "$MODEL" --exp_dir "$EXP_DIR" --temperature 1.0 \
    --save_file_name "tests" --raw_dataset_config "$DATASET_CONFIG" --seed 1
