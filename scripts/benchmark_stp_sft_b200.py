"""Search valid microbatches with full global-batch optimizer steps on one B200."""

import argparse
import csv
import importlib.metadata
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from RL.utils.experiment_utils import levanter_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='jobs/yaml/UCloud/sft_stp_paper_B200.yaml')
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    env = levanter_environment()
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    assert hardware.split(',')[0] == 'NVIDIA B200' and '\n' not in hardware, hardware
    (args.directory / 'hardware.json').write_text(json.dumps(dict(hardware=hardware, environment={
        key: env.get(key) for key in ('XLA_FLAGS', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                                     'XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB', 'CUDA_VISIBLE_DEVICES')}), indent=2))
    (args.directory / 'config.yaml').write_text(Path(args.config).read_text())
    (args.directory / 'versions.json').write_text(json.dumps({
        package: importlib.metadata.version(package)
        for package in ('jax', 'jaxlib', 'haliax', 'levanter', 'torch', 'vllm')}, indent=2))
    command = [sys.executable, '-u', '-m', 'scripts.benchmark_stp_sft_worker',
               '--config', args.config, '--directory', str(args.directory)]
    subprocess.run(command + ['--prepare'], env=env, check=True)
    rows = []

    def trial(batch):
        result_path = args.directory / f'batch-{batch}.json'
        env['BENCHMARK_RESULT'] = str(result_path)
        log_path = args.directory / f'batch-{batch}.log'
        peak = 0
        timed_out = False
        with log_path.open('w') as log:
            process = subprocess.Popen(command + ['--batch-size', str(batch)], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            while process.poll() is None:
                memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                                  '--format=csv,noheader,nounits'], text=True)
                peak = max(peak, int(memory.strip()))
                if time.monotonic() - start > 3600:
                    os.killpg(process.pid, signal.SIGKILL)
                    timed_out = True
                    break
                time.sleep(0.1)
            returncode = process.wait()
        log_text = log_path.read_text()
        status = 'success' if returncode == 0 else 'error'
        if timed_out:
            status = 'timeout'
        elif returncode != 0 and any(message in log_text for message in (
                'RESOURCE_EXHAUSTED', 'CUDA_ERROR_OUT_OF_MEMORY', 'Out of memory while trying to allocate')):
            status = 'oom'
        row = dict(mode='training', status=status, batch_size=batch, sequence_length=2048,
                   global_batch_size=2048, gradient_accumulation=2048 // batch,
                   measured_iterations=0, median_iteration_s=None, tokens_per_s=None,
                   peak_gpu_mib=peak, total_gpu_mib=int(hardware.split(',')[1]),
                   peak_live_gpu_mib=None, allocator_limit_mib=None,
                   memory_percent=100 * peak / int(hardware.split(',')[1]))
        if status == 'success':
            records = json.loads(result_path.read_text())
            measured = records[1:]
            assert len(measured) == 5
            row.update(measured_iterations=len(measured),
                       median_iteration_s=statistics.median(record['duration_s'] for record in measured),
                       tokens_per_s=2048 * 2048 / statistics.median(record['duration_s'] for record in measured),
                       peak_live_gpu_mib=max(record['memory_stats']['peak_bytes_in_use'] for record in records) / 2**20,
                       allocator_limit_mib=records[-1]['memory_stats']['bytes_limit'] / 2**20)
        rows.append(row)
        (args.directory / 'results.json').write_text(json.dumps(rows, indent=2))
        with (args.directory / 'results.csv').open('w') as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if status not in ('success', 'oom'):
            raise RuntimeError(f'{status}: {log_path}')
        return status == 'success'

    # Valid candidates are divisors of 2048: binary search over powers of two.
    lower = 0
    upper = 3
    if trial(8):
        lower = 3
        upper = 4
        while trial(2**upper):
            lower = upper
            upper += 1
            assert upper <= 11, 'Reached global batch without an OOM boundary'
    else:
        assert trial(1), 'Even batch 1 OOMs'
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if trial(2**middle):
            lower = middle
        else:
            upper = middle
    # Measure the next smaller size as a headroom/throughput alternative.
    alternative = 2**max(0, lower - 1)
    if alternative not in [row['batch_size'] for row in rows]:
        trial(alternative)
    (args.directory / 'boundary.json').write_text(json.dumps(dict(
        largest_success=2**lower, first_oom=2**upper, valid_sizes='divisors of global batch 2048'), indent=2))


if __name__ == '__main__':
    main()
