"""Score fixed reference-proof transitions from the 400 RL validation theorems."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from alphaproof.core.config import rl_config_from_dict
from alphaproof.core.network import Network
from alphaproof.training.sft import batch_losses
from scripts.prepare_deltaproof_validation import EXTRACTION, load_prepared, write_json, write_jsonl


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(config):
    source = Path(config['validation_data'])
    dataset = read_jsonl(source)
    held_out = read_jsonl(Path(config['validation_split']))
    assert len(dataset) == 400
    assert {r['id'] for r in dataset} <= {r['id'] for r in held_out}
    assert len({r['id'] for r in dataset}) == 400
    held_out_by_id = {r['id']: r['theorem'] for r in held_out}
    assert all(row['theorem'] == held_out_by_id[row['id']] for row in dataset)
    training_ids = {r['id'] for r in read_jsonl(Path(config['training_split']))}
    assert training_ids.isdisjoint(r['id'] for r in dataset)
    checkpoints = [dict(run='starting_model', round=None, game=0, checkpoint=config['starting_checkpoint'])]
    saved_configs = {}
    for name, directory in config['checkpoint_runs'].items():
        run = Path(directory)
        saved_configs[name] = json.loads((run / 'deltaproof/config_01.json').read_text())['config']
        games = 0
        for round_id in config['rounds']:
            directory = run / f'round{round_id}'
            assert (directory / 'round_complete').is_file()
            games += json.loads((directory / 'round_metrics.json').read_text())['attempts']
            checkpoints.append(dict(run=name, round=round_id, game=games,
                                    checkpoint=str(directory / 'solver_checkpoint.pt')))
    expected_shapes = None
    for checkpoint in checkpoints:
        params = torch.load(checkpoint['checkpoint'], map_location='meta', weights_only=True, mmap=True)
        weights = params if checkpoint['round'] is None else params['network_params']
        shapes = {key: tuple(value.shape) for key, value in weights.items()}
        if expected_shapes is None:
            expected_shapes = shapes
        assert shapes == expected_shapes, checkpoint['checkpoint']
        del params, weights
    search = next(iter(saved_configs.values()))
    for saved in saved_configs.values():
        for field in ['tokenizer_model', 'max_state_length', 'max_action_length',
                      'num_value_bins', 'value_weight', 'dtype']:
            assert saved[field] == search[field], field
    manifest = dict(config=config, checkpoints=checkpoints, solver_configs=saved_configs,
                    validation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    metric='reference-proof policy loss + value_weight * proof-depth value loss',
                    aggregation='transition-weighted mean of fixed batches, matching learner evaluation',
                    tokenization='fixed padding and truncation matching RL replay',
                    extraction=EXTRACTION,
                    states='single goals recorded during compilation of the original reference proofs',
                    actions='original tactic source; no rewriting, AST reconstruction, or replay',
                    value_targets='negative remaining source-tactic graph depth')
    return dataset, checkpoints, search, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    dataset, checkpoints, saved, manifest = prepare(config)
    output = Path(config['output_dir'])
    records, prepared = load_prepared(dataset, output, manifest['validation_sha256'])
    manifest['prepared'] = prepared
    tokenizer = AutoTokenizer.from_pretrained(saved['tokenizer_model'], local_files_only=True)
    observations = tokenizer([r['state'] for r in records], max_length=saved['max_state_length'],
                             padding='max_length', truncation=True, return_tensors='pt').input_ids
    actions = tokenizer([r['action'] for r in records], max_length=saved['max_action_length'],
                        padding='max_length', truncation=True, return_tensors='pt').input_ids
    values = torch.tensor([r['value'] for r in records], dtype=torch.float32)
    assert observations.shape == (len(records), saved['max_state_length'])
    assert actions.shape == (len(records), saved['max_action_length'])
    print(f'Validated {len(dataset)} held-out theorems; {len(records)} transitions; '
          f'{len(checkpoints)} checkpoint formats', flush=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Evaluation inputs changed.'
    if args.prepare_only:
        return
    assert torch.cuda.is_available(), 'Submit this job with a GPU.'
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    torch.set_num_threads(config['cpu_threads'])
    network = Network(rl_config_from_dict(saved, 'validation_loss'))
    summaries = []
    for checkpoint in checkpoints:
        name = f"{checkpoint['run']}_round{checkpoint['round']}"
        result_path = output / f'{name}.json'
        if not result_path.exists():
            started = time.time()
            params = torch.load(checkpoint['checkpoint'], map_location='cpu', weights_only=True)
            step = 0 if checkpoint['round'] is None else int(params['step'])
            network.params = params if checkpoint['round'] is None else params['network_params']
            del params
            network.eval()
            totals = [0.0, 0.0, 0.0]
            with torch.inference_mode():
                for start in range(0, len(records), config['batch_size']):
                    end = min(start + config['batch_size'], len(records))
                    batch = observations[start:end], actions[start:end], values[start:end]
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=network.mixed_precision):
                        loss, policy, value, result = batch_losses(network, batch, network.value_weight)
                    for i, metric in enumerate([loss, policy, value]):
                        totals[i] += metric.item() * (end - start)
                    del result
            metrics = dict(zip(['loss', 'policy_loss', 'value_loss'],
                               [total / len(records) for total in totals]))
            assert all(math.isfinite(value) for value in metrics.values())
            write_json(result_path, dict(checkpoint, **metrics, step=step,
                                         theorem_count=len(dataset), transitions=len(records),
                                         seconds=time.time() - started))
        summaries.append(json.loads(result_path.read_text()))
        write_jsonl(output / 'results.jsonl', summaries)
        print(f'COMPLETED {name}: {summaries[-1]}', flush=True)
    (output / 'SUCCESS').write_text(f'{len(summaries)} checkpoints completed\n')


if __name__ == '__main__':
    main()
