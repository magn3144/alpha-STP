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
from leantree import LeanProject, StoredError

from alphaproof.core.config import rl_config_from_dict
from alphaproof.core.network import Network
from alphaproof.training.sft import batch_losses


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


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
    for checkpoint in checkpoints:
        assert Path(checkpoint['checkpoint']).stat().st_size > 0
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
                    value_targets='negative remaining reference proof-tree depth')
    return dataset, checkpoints, search, manifest


def extract(dataset, project_path, output):
    """Cache each complete theorem; require coverage of all 400 theorems."""
    cache = output / 'theorems'
    cache.mkdir(exist_ok=True)
    sources = output / 'sources'
    sources.mkdir(exist_ok=True)
    project = LeanProject(project_path)
    records = []
    for index, row in enumerate(dataset):
        path = cache / f'{index:03d}.json'
        if not path.exists():
            source = sources / f'{index:03d}.lean'
            source.write_text(row['verified_proof'])
            lean_file = project.load_file(source, use_cache=False, store_assertion_errors=False)
            transitions = []
            for theorem in lean_file.theorems:
                assert not isinstance(theorem, StoredError), theorem
                for block in theorem.by_blocks:
                    assert block.tree is not None and not isinstance(block.tree, StoredError), block.tree
                    for node in block.tree.get_nodes():
                        if node.state is None or node.tactic is None:
                            continue
                        action = str(node.tactic.tactic).strip()
                        if 'sorry' in action or 'admit' in action or action == 'bound':
                            continue
                        transitions.append(dict(id=row['id'], state=str(node.state),
                                                action=action, value=-float(node.proof_depth)))
            assert transitions, row['id']
            write_json(path, dict(id=row['id'], transitions=transitions))
        cached = json.loads(path.read_text())
        assert cached['id'] == row['id']
        records.extend(cached['transitions'])
        print(f'Extracted {index + 1}/{len(dataset)} theorems; {len(records)} transitions', flush=True)
    assert {r['id'] for r in records} == {r['id'] for r in dataset}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    dataset, checkpoints, saved, manifest = prepare(config)
    print(f'{len(dataset)} held-out theorems; {len(checkpoints)} checkpoints', flush=True)
    if args.prepare_only:
        return
    assert torch.cuda.is_available(), 'Submit this job with a GPU.'
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Evaluation inputs changed.'
    else:
        write_json(manifest_path, manifest)
    records = extract(dataset, config['lean_project'], output)
    (output / 'transitions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    torch.set_num_threads(config['cpu_threads'])
    network = Network(rl_config_from_dict(saved, 'validation_loss'))
    observations = network.tokenizer([r['state'] for r in records],
                                    max_length=network.max_state_length,
                                    padding='max_length', truncation=True, return_tensors='pt').input_ids
    actions = network.tokenizer([r['action'] for r in records],
                               max_length=network.max_action_length,
                               padding='max_length', truncation=True, return_tensors='pt').input_ids
    values = torch.tensor([r['value'] for r in records], dtype=torch.float32)
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
        (output / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in summaries))
        print(f'COMPLETED {name}: {summaries[-1]}', flush=True)
    (output / 'SUCCESS').write_text(f'{len(summaries)} checkpoints completed\n')


if __name__ == '__main__':
    main()
