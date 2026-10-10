"""Smoke training, evaluation, and HF export with STP's actual model and microbatch."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from RL.utils.config_utils import load_experiment_config


def main(config_path, full_batch):
    config = load_experiment_config(config_path, 'sft')
    experiment = config['experiment']
    training = config['training']
    trainer = training['trainer']
    steps = 1 if full_batch else 2
    batch_size = trainer['train_batch_size'] if full_batch else 16
    run_id = 'smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    smoke_dir = Path(os.environ['DATA']) / 'runs' / 'smoke_stp_paper_b200' / run_id
    smoke_dir.mkdir(parents=True)
    for key, count in (('train_data', batch_size * steps), ('validation_data', trainer['per_device_eval_parallelism'])):
        with Path(experiment[key]).open() as source:
            rows = json.load(source)[:count]
        path = smoke_dir / f'{key}.json'
        path.write_text(json.dumps(rows))
        experiment[key] = str(path)
    experiment['run_dir'] = str(smoke_dir)
    trainer.update(train_batch_size=batch_size, num_train_steps=steps, steps_per_eval=1, max_eval_batches=1)
    trainer['tracker']['name'] = run_id
    training['save_freq'] = steps
    training['optimizer']['warmup'] = 0 if full_batch else 1
    path = smoke_dir / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['RAY_TMPDIR'] = '/tmp'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    env['XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB'] = '128'
    env['XLA_FLAGS'] = env.get('XLA_FLAGS', '') + ' --xla_gpu_enable_triton_gemm=false'
    subprocess.run(
        [sys.executable, '-u', 'RL/run_sft.py', '--config', str(path), '--run-id', run_id],
        env=env, check=True,
    )
    if full_batch:
        # Upstream skips both checkpoint callbacks at step zero.
        print(f'Full-batch SFT smoke passed: {smoke_dir}', flush=True)
        return
    checkpoint = smoke_dir / 'models' / run_id / f'step-{steps - 1}'
    assert (checkpoint / 'config.json').is_file(), checkpoint
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    assert all((checkpoint / shard).is_file() for shard in set(index['weight_map'].values()))
    native_checkpoint = smoke_dir / 'checkpoints' / run_id / f'step-{steps - 1}'
    assert (native_checkpoint / 'metadata.json').is_file(), native_checkpoint
    print(f'SFT smoke passed: {checkpoint}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--full-batch', action='store_true',
                        help='Run one optimizer step with the full global batch from the config.')
    args = parser.parse_args()
    main(args.config, args.full_batch)
