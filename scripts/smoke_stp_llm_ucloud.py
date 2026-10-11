"""Run two small STP rounds with the production model, verifier, and trainer."""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from RL.utils.config_utils import load_experiment_config
from RL.utils.file_utils import read_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='jobs/yaml/UCloud/rl_llm_stp_B200.yaml')
    parser.add_argument('--generation-only', action='store_true',
                        help='Verify generation, Lean, and conjecture selection without training on small MIG slices.')
    parser.add_argument('--check-run', type=Path,
                        help='Check an existing completed smoke run without launching training.')
    args = parser.parse_args()
    if args.check_run:
        directory = args.check_run.resolve()
    else:
        config = load_experiment_config(args.config, 'rl')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = Path(os.environ['DATA']) / 'runs' / 'smoke_stp_llm_b200' / run_id
        directory.mkdir(parents=True)
        experiment = config['experiment']
        experiment.update(exp_dir=str(directory), total_rounds=2, dataset_size=2048)
        experiment['llm']['attempts_per_round'] = 2048
        if args.generation_only:
            experiment['epochs'] = 0
            experiment['dataset_size'] = 128
            experiment['llm']['attempts_per_round'] = 64
        config['training']['trainer']['tracker']['name'] = f'stp-smoke-{run_id}'
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run([sys.executable, '-u', 'RL/run_rl_steps.py', '--config', str(path)], check=True)
    summary = []
    for round_id in range(2):
        round_dir = directory / f'round{round_id}'
        assert (round_dir / 'round_complete').is_file(), round_dir
        if args.generation_only:
            proofs = read_file(str(round_dir / 'sampler_ckpt/generated_proofs.json'))
            assert proofs, round_dir
            assert any(proof.get('complete', False) for proof in proofs), round_dir
            conjectures = [proof for proof in proofs
                           if any(label.startswith('conjecture ') for label in proof['label'])]
            if round_id == 1:
                assert conjectures, round_dir
            summary.append(dict(round=round_id, generated_proofs=len(proofs),
                                verified_proofs=sum(proof.get('complete', False) for proof in proofs),
                                conjecture_attempts=sum(proof.get('multiplicity', 1) for proof in conjectures)))
            continue
        model = round_dir / 'RL_model'
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        assert all((model / shard).stat().st_size > 0 for shard in set(index['weight_map'].values()))
        assert (model / 'tokenizer.json').is_file(), model
        dataset = json.loads((round_dir / 'train_ds.json').read_text())
        summary.append(dict(round=round_id, training_examples=len(dataset),
                            optimizer_steps=max(5, len(dataset) // 32), model=str(model)))
    if not args.generation_only:
        training_losses = {}
        wandb_dir = Path(os.environ['WANDB_DIR']) / 'wandb'
        for path in sorted(wandb_dir.glob('run-*/files/wandb-summary.json')):
            metrics = json.loads(path.read_text())
            if 'train/loss' in metrics:
                run_config = yaml.safe_load((path.parent / 'config.yaml').read_text())
                train_data = Path(run_config['train_data']['value'])
                if train_data.parent.parent != directory:
                    continue
                assert math.isfinite(metrics['train/loss']), path
                training_losses[train_data.parent.name] = metrics['train/loss']
        assert set(training_losses) == {'round0', 'round1'}, training_losses
        summary.append(dict(final_training_losses=training_losses))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    mode = 'generation/verification' if args.generation_only else 'generation/verification/training'
    print(f'Two-round STP {mode} smoke passed: {directory}', flush=True)


if __name__ == '__main__':
    main()
