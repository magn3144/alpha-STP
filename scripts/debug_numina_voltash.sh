#!/bin/sh
set -eu

: "${LSF_QRSH:?Run this script inside a voltash session.}"
: "${LSB_SUB_HOST:?The voltash session does not expose its submission host.}"

set -- "$HOME"/.vscode-server/extensions/ms-python.debugpy-*/bundled/libs
if [ "$#" -ne 1 ] || [ ! -d "$1/debugpy" ]; then
    echo "Expected exactly one VS Code debugpy extension installation." >&2
    exit 1
fi
debugpy_root=$1

ssh -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -R 127.0.0.1:5678:127.0.0.1:5678 \
    "$LSB_SUB_HOST" &
tunnel_pid=$!

cleanup() {
    if kill -0 "$tunnel_pid" 2>/dev/null; then
        kill "$tunnel_pid"
        wait "$tunnel_pid" || true
    fi
}
trap cleanup EXIT INT TERM

sleep 1
if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    wait "$tunnel_pid"
fi

export PYTHONPATH="$debugpy_root"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$PWD/.venv/bin/python" -Xfrozen_modules=off -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    --configure-subProcess true \
    -m stp.evaluate_difficulty_score.evaluate_numina \
    --config configs/debug_numina.toml \
    --problem-index 8
