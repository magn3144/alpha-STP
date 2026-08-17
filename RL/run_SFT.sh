#!/bin/bash

set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set by LSF}"
: "${STORAGE:?Set STORAGE to a local DTU filesystem path}"

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/levanter:$REPO_DIR/levanter/src:$REPO_DIR/levanter/examples${PYTHONPATH:+:$PYTHONPATH}"
python -u levanter/examples/weighted_lm.py \
    --config_path levanter/config/sft.yaml \
    --trainer.checkpointer.base_path "$STORAGE/SFT_ckpt" \
    --hf_save_path "$STORAGE/SFT" \
    --train_data "$STORAGE/data/SFT/mathlib_leanworkbook.json" \
    --train_data_cache_dir "$STORAGE/data/SFT/mathlib_leanworkbook_cache" \
    --eval_data "$STORAGE/data/SFT/eval.json" \
    --eval_data_cache_dir "$STORAGE/data/SFT/eval_cache"
