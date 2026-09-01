import argparse
import csv
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


OOM_MARKERS = (
    "out of memory",
    "resource_exhausted",
    "cuda_error_out_of_memory",
    "failed to allocate",
)


def gpu_inventory():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    rows = []
    for line in subprocess.check_output(command, text=True).strip().splitlines():
        name, memory, driver = [value.strip() for value in line.split(",")]
        rows.append({"name": name, "memory_mib": int(memory), "driver": driver})
    return rows


def validate_gpus(gpus):
    if len(gpus) != 2:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}: {gpus}")
    for gpu in gpus:
        if "A100" not in gpu["name"]:
            raise RuntimeError(f"Expected A100 GPUs, found {gpu['name']}")
        if not 39000 <= gpu["memory_mib"] <= 42000:
            raise RuntimeError(f"Expected 40 GB A100 GPUs, found {gpu['memory_mib']} MiB")


def sample_memory(stop_event, peaks):
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            values = [int(value.strip()) for value in subprocess.check_output(command, text=True).splitlines()]
            for index, value in enumerate(values):
                peaks[index] = max(peaks[index], value)
        except subprocess.SubprocessError:
            pass
        stop_event.wait(0.1)


def read_records(path):
    if not path.exists():
        return []
    with path.open() as record_file:
        return [json.loads(line) for line in record_file if line.strip()]


def run_trial(args, batch_size, gpu_count, total_gpu_mib):
    trial_dir = args.output_dir / f"batch_{batch_size}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    performance_log = trial_dir / "steps.jsonl"
    process_log = trial_dir / "process.log"
    checkpoint_dir = trial_dir / "checkpoints"

    environment = os.environ.copy()
    python_paths = [args.repo / "levanter", args.repo / "levanter/src", args.repo / "levanter/examples"]
    if environment.get("PYTHONPATH"):
        python_paths.append(Path(environment["PYTHONPATH"]))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    environment["LEVANTER_BENCHMARK_LOG"] = str(performance_log)
    environment["LEVANTER_BENCHMARK_ITERATIONS"] = str(args.warmup_iterations + args.measured_iterations)
    environment["WANDB_MODE"] = "disabled"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    per_device_batch = batch_size // gpu_count
    command = [
        sys.executable,
        "-u",
        str(args.repo / "levanter/examples/weighted_lm.py"),
        "--config_path",
        str(args.config),
        "--model_name_or_path",
        str(args.model),
        "--tokenizer_name_or_path",
        str(args.model),
        "--train_data",
        str(args.train_data),
        "--train_data_cache_dir",
        str(args.train_cache),
        "--trainer.train_batch_size",
        str(batch_size),
        "--trainer.per_device_parallelism",
        str(per_device_batch),
        "--trainer.per_device_eval_parallelism",
        str(per_device_batch),
        "--trainer.checkpointer.base_path",
        str(checkpoint_dir),
    ]

    peaks = [0] * gpu_count
    stop_event = threading.Event()
    monitor = threading.Thread(target=sample_memory, args=(stop_event, peaks))
    monitor.start()
    timed_out = False
    with process_log.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=args.trial_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return_code = process.returncode
    stop_event.set()
    monitor.join()

    records = read_records(performance_log)
    measured = records[args.warmup_iterations:]
    log_text = process_log.read_text(errors="replace").lower()

    if timed_out:
        status = "timeout"
    elif return_code != 0:
        status = "oom" if any(marker in log_text for marker in OOM_MARKERS) else "error"
    elif len(measured) != args.measured_iterations:
        status = "error"
    elif not all(math.isfinite(record["loss"]) and math.isfinite(record["duration_s"]) for record in measured):
        status = "error"
    else:
        status = "success"

    durations = [record["duration_s"] for record in measured]
    token_rates = [record["tokens_per_s"] for record in measured]
    peak_gpu_mib = max(peaks)
    result = {
        "mode": "training",
        "status": status,
        "batch_size": batch_size,
        "per_device_batch_size": per_device_batch,
        "sequence_length": 2048,
        "warmup_iterations": min(len(records), args.warmup_iterations),
        "measured_iterations": len(measured),
        "median_iteration_s": statistics.median(durations) if durations else "",
        "tokens_per_s": statistics.median(token_rates) if token_rates else "",
        "peak_gpu_mib": peak_gpu_mib,
        "total_gpu_mib": total_gpu_mib,
        "memory_percent": peak_gpu_mib / total_gpu_mib * 100,
        "median_loss": statistics.median(record["loss"] for record in measured) if measured else "",
        "log": str(process_log),
    }
    print(
        f"{status}\tbatch={batch_size}\tmedian_s={result['median_iteration_s']}\t"
        f"tokens_s={result['tokens_per_s']}\tpeak_mib={peak_gpu_mib}",
        flush=True,
    )
    return result


def write_results(path, results):
    fields = [
        "mode",
        "status",
        "batch_size",
        "per_device_batch_size",
        "sequence_length",
        "warmup_iterations",
        "measured_iterations",
        "median_iteration_s",
        "tokens_per_s",
        "peak_gpu_mib",
        "total_gpu_mib",
        "memory_percent",
        "median_loss",
        "log",
    ]
    with path.open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)


def main(args):
    args.repo = args.repo.resolve()
    args.model = args.model.resolve()
    args.config = args.config.resolve()
    args.train_data = args.train_data.resolve()
    args.train_cache = args.train_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpus = gpu_inventory()
    validate_gpus(gpus)
    total_gpu_mib = min(gpu["memory_mib"] for gpu in gpus)
    metadata = {
        "job_id": os.environ["LSB_JOBID"],
        "mode": "training",
        "model": str(args.model),
        "config": str(args.config),
        "sequence_length": 2048,
        "precision": "p=f32,c=bfloat16",
        "optimizer": "Adam",
        "gradient_checkpointing": "Levanter default",
        "attention_implementation": "Levanter default",
        "sharding": "FSDP over 2 GPUs with tensor-parallel mlp and heads axes",
        "data": str(args.train_data),
        "python": platform.python_version(),
        "gpus": gpus,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    gpu_count = len(gpus)
    upper_per_device = args.upper_global_batch // gpu_count
    results = []
    attempted = {}

    def attempt(per_device_batch):
        batch_size = per_device_batch * gpu_count
        if batch_size not in attempted:
            result = run_trial(args, batch_size, gpu_count, total_gpu_mib)
            attempted[batch_size] = result
            results.append(result)
            write_results(args.output_dir / "results.tsv", results)
        return attempted[batch_size]

    per_device_batch = 1
    largest_success = None
    first_oom = None
    while per_device_batch <= upper_per_device:
        result = attempt(per_device_batch)
        if result["status"] == "success":
            largest_success = per_device_batch
            per_device_batch *= 2
            continue
        if result["status"] == "oom":
            first_oom = per_device_batch
            break
        raise RuntimeError(f"Batch {result['batch_size']} failed with {result['status']}; see {result['log']}")

    if first_oom is not None and largest_success is not None:
        low = largest_success
        high = first_oom
        while high - low > 1:
            middle = (low + high) // 2
            result = attempt(middle)
            if result["status"] == "success":
                low = middle
            elif result["status"] == "oom":
                high = middle
            else:
                raise RuntimeError(f"Batch {result['batch_size']} failed with {result['status']}; see {result['log']}")
        largest_success = low
        first_oom = high

    successes = [result for result in results if result["status"] == "success"]
    if not successes:
        raise RuntimeError("No batch size completed successfully")
    headroom = [result for result in successes if result["memory_percent"] <= 90]
    recommended = max(headroom or successes, key=lambda result: result["batch_size"])
    largest = max(successes, key=lambda result: result["batch_size"])
    summary = {
        "largest_successful_global_batch": largest["batch_size"],
        "first_oom_global_batch": first_oom * gpu_count if first_oom is not None else None,
        "tested_upper_global_batch": args.upper_global_batch,
        "recommended_global_batch": recommended["batch_size"],
        "recommended_memory_percent": recommended["memory_percent"],
        "recommended_tokens_per_s": recommended["tokens_per_s"],
        "results": str(args.output_dir / "results.tsv"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upper-global-batch", type=int, default=64)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=5)
    parser.add_argument("--trial-timeout", type=int, default=1200)
    main(parser.parse_args())
