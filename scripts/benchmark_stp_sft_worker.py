"""Run one full-length B200 candidate through the original SFT training path."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import draccus
import jax
import transformers
import weighted_lm

from RL.utils.config_utils import load_experiment_config


class BenchmarkTrainer(weighted_lm.Trainer):
    def train(self, state, train_loader, run_hooks=True):
        records = []
        loader = iter(train_loader)
        for step in range(6):
            batch = next(loader)
            jax.block_until_ready((state, batch))
            start = time.monotonic()
            info = self.train_step(state, batch)
            state = info.state
            jax.block_until_ready(state)
            duration = time.monotonic() - start
            assert math.isfinite(info.loss), info.loss
            record = dict(step=step, warmup=step == 0, duration_s=duration,
                          loss=info.loss, memory_stats=jax.devices()[0].memory_stats())
            records.append(record)
            Path(os.environ['BENCHMARK_RESULT']).write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
        return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    experiment_config = load_experiment_config(args.config, 'sft')
    experiment = experiment_config['experiment']
    training = experiment_config['training']
    length = training['max_tune_length']
    data_path = args.directory / 'full_length.json'
    if args.prepare:
        tokenizer = transformers.AutoTokenizer.from_pretrained(experiment['base_model'])
        rows = json.loads(Path(experiment['train_data']).read_text())
        row = next(row for row in rows if len(tokenizer(row['prompt'] + row['target'])['input_ids']) >= length)
        # Original preprocessing truncates this real sample to exactly max_tune_length.
        data_path.write_text(json.dumps([row] * training['trainer']['train_batch_size']))
        (args.directory / 'sample.json').write_text(json.dumps(dict(
            sequence_length=length, untruncated_tokens=len(tokenizer(row['prompt'] + row['target'])['input_ids']))))
        return
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].device_kind == 'NVIDIA B200', devices
    trainer = training['trainer']
    trainer.update(per_device_parallelism=args.batch_size, tracker={'type': 'noop'},
                   id=f'batch-{args.batch_size}', load_checkpoint=False,
                   checkpointer={'base_path': str(args.directory / 'unused-checkpoints'),
                                 'save_interval': '0s', 'keep': []})
    training.update(model_name_or_path=experiment['base_model'], tokenizer_name_or_path=experiment['base_model'],
                    train_data=str(data_path), train_data_cache_dir=str(args.directory / 'cache'),
                    eval_data=None, hf_save_path=None)
    config = draccus.decode(weighted_lm.TrainArgs, training)
    weighted_lm.Trainer = BenchmarkTrainer
    weighted_lm.train(config)


if __name__ == '__main__':
    main()
