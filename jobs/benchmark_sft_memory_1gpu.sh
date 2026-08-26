#!/bin/bash
#BSUB -J stp-sft-memory-1gpu
#BSUB -q gpua100
#BSUB -W 2:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -oo jobs/logs/sft_memory_1gpu_%J.out
#BSUB -eo jobs/logs/sft_memory_1gpu_%J.err

set -uo pipefail

source "$LS_SUBCWD/jobs/common.sh"
: "${BASE_MODEL:?Set BASE_MODEL in jobs/config.sh}"
: "${LSB_JOBID:?Run this script through bsub}"
set +e

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off

model="${BENCHMARK_MODEL:-$BASE_MODEL}"
train_data="$STORAGE/data/SFT/train.json"
train_cache="$STORAGE/data/SFT/train_cache"
result_dir="$STORAGE/SFT_memory_benchmark_1gpu/$LSB_JOBID"
max_batch="${MAX_BATCH:-32}"
monitor_pid=

mkdir -p "$result_dir"
summary="$result_dir/summary.tsv"
printf 'status\tbatch\tpeak_gpu_mib\n' > "$summary"

cleanup_monitor() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
}
trap cleanup_monitor EXIT INT TERM

run_trial() {
    batch=$1
    trial_dir="$result_dir/batch_$batch"
    memory_log="$trial_dir/memory.csv"
    train_log="$trial_dir/train.log"

    mkdir -p "$trial_dir"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        --loop-ms=100 --filename="$memory_log" &
    monitor_pid=$!

    echo "Testing one-GPU batch $batch"
    python -u levanter/examples/weighted_lm.py \
        --config_path levanter/config/sft.yaml \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --trainer.num_train_steps 1 \
        --optimizer.warmup 0 \
        --trainer.train_batch_size "$batch" \
        --trainer.per_device_parallelism "$batch" \
        --trainer.per_device_eval_parallelism "$batch" \
        --trainer.load_checkpoint false \
        --trainer.checkpointer.base_path "$trial_dir/checkpoints" \
        --hf_save_path null \
        --eval_data null \
        --train_data "$train_data" \
        --train_data_cache_dir "$train_cache" \
        > "$train_log" 2>&1
    status=$?

    cleanup_monitor
    peak_gpu=$(awk -F, '{value=$1 + 0; if (value > peak) peak=value} END {print peak + 0}' "$memory_log")

    if [[ "$status" -eq 0 ]]; then
        printf 'success\t%s\t%s\n' "$batch" "$peak_gpu" | tee -a "$summary"
        return 0
    fi

    if rg -qi 'out of memory|resource_exhausted|failed to allocate' "$train_log"; then
        printf 'oom\t%s\t%s\n' "$batch" "$peak_gpu" | tee -a "$summary"
        return 2
    fi

    printf 'error\t%s\t%s\n' "$batch" "$peak_gpu" | tee -a "$summary"
    return 1
}

last_success=0
first_oom=0
candidate=1

while (( candidate <= max_batch )); do
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
    echo "Maximum not reached: batch $last_success succeeded." | tee "$result_dir/result.txt"
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
    echo "Maximum successful one-GPU batch: $last_success"
    echo "First known OOM batch: $first_oom"
    echo "Results: $summary"
} | tee "$result_dir/result.txt"
