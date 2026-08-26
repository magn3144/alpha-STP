#!/bin/bash
#BSUB -J stp-sft-memory-l40s
#BSUB -q gpul40s
#BSUB -W 2:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -oo jobs/logs/sft_memory_l40s_%J.out
#BSUB -eo jobs/logs/sft_memory_l40s_%J.err

set -uo pipefail

source "$LS_SUBCWD/jobs/common.sh"
: "${LSB_JOBID:?Run this script through bsub}"
set +e

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off

model="${BENCHMARK_MODEL:-$STORAGE/models/deepseek-coder-1.3b-base}"
train_data="$STORAGE/data/SFT/train.json"
train_cache="$STORAGE/data/SFT/train_cache"
result_dir="$STORAGE/SFT_memory_benchmark_l40s/$LSB_JOBID"
max_batch="${MAX_BATCH:-32}"
monitor_pid=

mkdir -p "$result_dir"
summary="$result_dir/summary.tsv"
printf 'status\tbatch\tpeak_gpu_mib\ttotal_gpu_mib\n' > "$summary"

if [[ ! -f "$train_data" ]]; then
    echo "Missing SFT training data: $train_data" >&2
    exit 1
fi

if [[ ! -d "$model" ]]; then
    echo "Missing model: $model" >&2
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

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
total_gpu=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
if [[ "$gpu_count" -ne 1 ]] || [[ "$gpu_name" != *L40S* ]]; then
    echo "Expected exactly one NVIDIA L40S; found $gpu_count GPU(s): $gpu_name" >&2
    exit 1
fi

echo "GPU: $gpu_name (${total_gpu} MiB)"
echo "Model: $model"
echo "Sequence length: 2048"

run_trial() {
    batch=$1
    trial_dir="$result_dir/batch_$batch"
    memory_log="$trial_dir/memory.csv"
    train_log="$trial_dir/train.log"

    mkdir -p "$trial_dir"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        --loop-ms=100 --filename="$memory_log" &
    monitor_pid=$!

    echo "Testing one-L40S batch $batch"
    python -u levanter/examples/weighted_lm.py \
        --config_path levanter/config/sft.yaml \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --trainer.num_train_steps 1 \
        --optimizer.warmup 0 \
        --trainer.train_batch_size "$batch" \
        --trainer.per_device_parallelism "$batch" \
        --trainer.per_device_eval_parallelism "$batch" \
        --trainer.ray.auto_start_cluster false \
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
        printf 'success\t%s\t%s\t%s\n' "$batch" "$peak_gpu" "$total_gpu" | tee -a "$summary"
        return 0
    fi

    if rg -qi 'out of memory|resource_exhausted|cuda_error_out_of_memory|failed to allocate' "$train_log"; then
        printf 'oom\t%s\t%s\t%s\n' "$batch" "$peak_gpu" "$total_gpu" | tee -a "$summary"
        return 2
    fi

    printf 'error\t%s\t%s\t%s\n' "$batch" "$peak_gpu" "$total_gpu" | tee -a "$summary"
    echo "Trial failed for a reason other than GPU OOM. See $train_log" >&2
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

if [[ "$first_oom" -ne 0 ]]; then
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
fi

safe_batch=0
while IFS=$'\t' read -r status batch peak total; do
    if [[ "$status" == success ]] && (( peak * 100 <= total * 90 )) && (( batch > safe_batch )); then
        safe_batch=$batch
    fi
done < "$summary"

{
    echo "Maximum successful batch: $last_success"
    if [[ "$first_oom" -eq 0 ]]; then
        echo "OOM boundary not reached; increase MAX_BATCH beyond $max_batch."
    else
        echo "First OOM batch: $first_oom"
    fi
    echo "Safe batch at <=90% measured GPU memory: $safe_batch"
    echo "Results: $summary"
} | tee "$result_dir/result.txt"
