#!/bin/bash
#BSUB -J stp-sft-memory
#BSUB -q gpua100
#BSUB -W 4:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -oo jobs/logs/sft_memory_%J.out
#BSUB -eo jobs/logs/sft_memory_%J.err

set -uo pipefail

source "$LS_SUBCWD/jobs/common.sh"
: "${BASE_MODEL:?Set BASE_MODEL in jobs/config.sh}"
: "${LSB_JOBID:?Run this script through bsub}"
set +e

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off

model="${BENCHMARK_MODEL:-$BASE_MODEL}"
train_data="$STORAGE/data/SFT/mathlib_leanworkbook.json"
train_cache="$STORAGE/data/SFT/mathlib_leanworkbook_cache"
result_dir="$STORAGE/SFT_memory_benchmark/$LSB_JOBID"
max_per_device_batch="${MAX_PER_DEVICE_BATCH:-64}"
monitor_pid=

mkdir -p "$result_dir"

if [[ ! -f "$train_data" ]]; then
    echo "Missing SFT training data: $train_data" >&2
    exit 1
fi

cleanup_monitor() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
}
trap cleanup_monitor EXIT INT TERM

gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l)
smallest_gpu=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if echo "$gpu_names" | rg -qv "NVIDIA A100"; then
    echo "Allocated GPUs are not both A100s:" >&2
    echo "$gpu_names" >&2
    exit 1
fi
if [[ "$gpu_count" -ne 2 || "$smallest_gpu" -lt 80000 ]]; then
    echo "Expected exactly two 80 GB A100s; found $gpu_count GPUs with minimum ${smallest_gpu} MiB." >&2
    exit 1
fi

python - <<'PY'
import jax

devices = jax.devices()
print(f"JAX devices: {devices}")
if len(devices) != 2:
    raise RuntimeError(f"Expected two JAX devices, found {len(devices)}")
PY
if [[ "$?" -ne 0 ]]; then
    exit 1
fi

summary="$result_dir/summary.tsv"
printf 'status\tper_device_batch\tglobal_microbatch\tpeak_gpu0_mib\tpeak_gpu1_mib\n' > "$summary"

run_trial() {
    per_device_batch=$1
    global_microbatch=$((2 * per_device_batch))
    trial_dir="$result_dir/per_device_$per_device_batch"
    memory_log="$trial_dir/memory.csv"
    train_log="$trial_dir/train.log"

    mkdir -p "$trial_dir"
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        --loop-ms=100 \
        --filename="$memory_log" &
    monitor_pid=$!

    echo "Testing per-device batch $per_device_batch (global microbatch $global_microbatch)"
    python -u levanter/examples/weighted_lm.py \
        --config_path levanter/config/sft.yaml \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --trainer.num_train_steps 1 \
        --trainer.train_batch_size "$global_microbatch" \
        --trainer.per_device_parallelism "$per_device_batch" \
        --trainer.per_device_eval_parallelism "$per_device_batch" \
        --trainer.mp p=bfloat16,c=bfloat16 \
        --trainer.load_checkpoint false \
        --trainer.checkpointer.base_path "$trial_dir/checkpoints" \
        --trainer.tracker.type noop \
        --hf_save_path null \
        --eval_data null \
        --train_data "$train_data" \
        --train_data_cache_dir "$train_cache" \
        > "$train_log" 2>&1
    status=$?

    cleanup_monitor
    peak_gpu0=$(awk -F, '$2 + 0 == 0 {value=$3 + 0; if (value > peak) peak=value} END {print peak + 0}' "$memory_log")
    peak_gpu1=$(awk -F, '$2 + 0 == 1 {value=$3 + 0; if (value > peak) peak=value} END {print peak + 0}' "$memory_log")

    if [[ "$status" -eq 0 ]]; then
        printf 'success\t%s\t%s\t%s\t%s\n' \
            "$per_device_batch" "$global_microbatch" "$peak_gpu0" "$peak_gpu1" | tee -a "$summary"
        return 0
    fi

    if rg -qi 'out of memory|resource_exhausted|cuda_error_out_of_memory|failed to allocate' "$train_log"; then
        printf 'oom\t%s\t%s\t%s\t%s\n' \
            "$per_device_batch" "$global_microbatch" "$peak_gpu0" "$peak_gpu1" | tee -a "$summary"
        return 2
    fi

    printf 'error\t%s\t%s\t%s\t%s\n' \
        "$per_device_batch" "$global_microbatch" "$peak_gpu0" "$peak_gpu1" | tee -a "$summary"
    echo "Trial failed for a reason other than GPU OOM. See $train_log" >&2
    return 1
}

last_success=0
first_oom=0
candidate=1

while (( candidate <= max_per_device_batch )); do
    run_trial "$candidate"
    result=$?
    if [[ "$result" -eq 0 ]]; then
        last_success=$candidate
        candidate=$((candidate * 2))
    elif [[ "$result" -eq 2 ]]; then
        first_oom=$candidate
        break
    else
        exit 1
    fi
done

if [[ "$first_oom" -eq 0 ]]; then
    echo "Maximum not reached: per-device batch $last_success succeeded." | tee "$result_dir/result.txt"
    echo "Increase MAX_PER_DEVICE_BATCH and resubmit to search higher." | tee -a "$result_dir/result.txt"
    exit 0
fi

lower=$((last_success + 1))
upper=$((first_oom - 1))
while (( lower <= upper )); do
    candidate=$(((lower + upper) / 2))
    run_trial "$candidate"
    result=$?
    if [[ "$result" -eq 0 ]]; then
        last_success=$candidate
        lower=$((candidate + 1))
    elif [[ "$result" -eq 2 ]]; then
        first_oom=$candidate
        upper=$((candidate - 1))
    else
        exit 1
    fi
done

{
    echo "Maximum successful per-device batch: $last_success"
    echo "Maximum successful global microbatch: $((2 * last_success))"
    echo "First known OOM per-device batch: $first_oom"
    echo "Results: $summary"
} | tee "$result_dir/result.txt"
