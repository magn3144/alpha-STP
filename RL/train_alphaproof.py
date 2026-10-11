"""STP round training using AlphaProof's policy/value network and checkpoints."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from alphaproof.core.config import load_experiment_config, rl_config_from_dict
from alphaproof.core.network import Network
from alphaproof.training.randomness import seed_everything
from alphaproof.training.run_config import changed_config_fields, has_run_config, load_run_config, save_run_config
from alphaproof.training.run_logger import RunLogger, initialize_stp_wandb, load_stp_wandb_settings
from alphaproof.training.shared_storage import SharedStorage
from alphaproof.training.sft import load_examples
from RL.utils.file_utils import read_file
from RL.utils.training_data_utils import select_training_proofs


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def encode_json(data):
    return (json.dumps(data, sort_keys=True, indent=2) + '\n').encode()


def save_progress(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encode_json(data))
    temporary.replace(path)


def immutable_file(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Inputs changed for an already prepared round: {path}')
    else:
        with path.open('xb') as output:
            output.write(data)


def prepare_dataset(round_dir):
    """Join selected whole proofs to their unmodified search transitions."""
    round_id = int(round_dir.name.removeprefix('round'))
    retained_rounds = list(range(max(0, round_id - 2), round_id + 1))
    proofs = read_file(str(round_dir / 'generated_proofs.json'))
    if proofs is None:
        raise FileNotFoundError(round_dir / 'generated_proofs.json')
    proofs = [proof for proof in proofs if proof['round'] in retained_rounds]
    selected = select_training_proofs(proofs)
    by_request = defaultdict(list)
    sources = {'proofs': fingerprint(encode_json(proofs))}
    for retained in retained_rounds:
        path = round_dir.parent / f'round{retained}' / 'deltaproof_transitions.jsonl'
        raw = path.read_bytes()
        sources[str(retained)] = fingerprint(raw)
        for line in raw.splitlines():
            record = json.loads(line)
            if record['batch_id'] != f'round{retained}':
                raise ValueError(f'Incorrect transition round in {path}')
            by_request[record['request_id']].append(record)
    transitions = []
    for proof in selected:
        records = by_request[proof['request_id']]
        if len(records) != proof['transition_count']:
            raise ValueError(f'Missing transitions for selected proof {proof["request_id"]}')
        transitions.extend(records)
    ids = [record['transition_id'] for record in transitions]
    if len(ids) != len(set(ids)):
        raise ValueError('Selected proofs have duplicate transition IDs.')
    snapshot = b''.join((json.dumps(record, sort_keys=True) + '\n').encode() for record in transitions)
    return transitions, snapshot, {
        'retained_rounds': retained_rounds,
        'selected_proofs': len(selected),
        'selected_request_ids': [proof['request_id'] for proof in selected],
        'sources': sources,
        'fingerprint': fingerprint(snapshot),
        'example_count': len(transitions),
    }


def tokenize(record, tokenizer, config):
    state = tokenizer(record['state'], max_length=config.max_state_length,
                      padding='max_length', truncation=True, return_tensors='pt').input_ids.squeeze(0).long()
    action = tokenizer(record['action'], max_length=config.max_action_length,
                       padding='max_length', truncation=True, return_tensors='pt').input_ids.squeeze(0).long()
    return state, action, record['value']


def evaluate(network, examples, tokenizer, config):
    total = 0.0
    for start in range(0, len(examples), config.validation_batch_size):
        batch = [tokenize({'state': example.state, 'action': example.action,
                           'value': example.value_target}, tokenizer, config)
                 for example in examples[start:start + config.validation_batch_size]]
        total += network.evaluate(batch) * len(batch)
    return total / len(examples)


def train_round(args, config, sft_config, epochs, logger):
    transitions, snapshot, manifest = prepare_dataset(args.round_dir)
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    if config.batch_size < 1 or config.gradient_accumulation_steps < 1 or epochs < 1:
        raise ValueError('Batch size, accumulation and learner epochs must be positive.')
    if len(transitions) < effective_batch:
        raise ValueError(f'Solver has {len(transitions)} transitions; one full batch requires {effective_batch}.')
    num_steps = max(5, len(transitions) * epochs // effective_batch)
    manifest |= {
        'effective_batch': effective_batch,
        'epochs': epochs,
        'calculated_steps': num_steps,
        'config_fingerprint': fingerprint(args.config.read_bytes()),
        'validation_fingerprint': fingerprint(sft_config.validation_input.read_bytes()),
    }
    snapshot_path = args.round_dir / 'solver_training_transitions.jsonl'
    manifest_path = args.round_dir / 'solver_training_manifest.json'
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing is not None:
        manifest['starting_solver_step'] = existing['starting_solver_step']
        if manifest != existing:
            raise ValueError('Inputs changed for an already prepared solver round.')
    immutable_file(snapshot_path, snapshot)
    progress_path = args.round_dir / 'solver_training_progress.json'
    target = existing['starting_solver_step'] + num_steps if existing else None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['complete']:
            print(f'{args.round_dir.name}: all {num_steps} solver updates already completed.', flush=True)
            return

    seed_everything(config.seed)
    storage = SharedStorage(args.run_dir)
    network = Network(config)
    if (storage.checkpoints_dir / 'latest.pt').exists():
        step = storage.load_latest_checkpoint(network)
    else:
        if existing is not None:
            raise ValueError('Prepared round has no optimizer checkpoint.')
        network.load_params(config.initial_params_path)
        step = 0
        storage.save_latest_checkpoint(step, network)
    if existing is None:
        manifest['starting_solver_step'] = step
        immutable_file(manifest_path, encode_json(manifest))
        target = step + num_steps
    start_step = manifest['starting_solver_step']
    if not start_step <= step <= target:
        raise ValueError('Latest checkpoint is outside this round\'s step budget.')
    print(f'STP solver training: rounds={manifest["retained_rounds"]}, '
          f'proofs={manifest["selected_proofs"]}, transitions={len(transitions)}, '
          f'steps={num_steps}, starting_step={start_step}', flush=True)
    logger.wandb_run.log({
        'stp/round': int(args.round_dir.name.removeprefix('round')),
        'stp/retained_rounds': manifest['retained_rounds'],
        'stp/selected_proofs': manifest['selected_proofs'],
        'stp/transitions': len(transitions), 'stp/training_steps': num_steps,
        'stp/effective_batch': effective_batch,
    })
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
    validation, stats = load_examples(sft_config.validation_input, tokenizer,
                                     config.max_state_length, config.max_action_length, None)
    if not validation:
        raise ValueError('The fixed SFT validation split contains no eligible transitions.')
    print(f'Fixed SFT validation: {stats}', flush=True)
    random.Random(config.seed + int(args.round_dir.name.removeprefix('round'))).shuffle(transitions)
    full_batches = len(transitions) // effective_batch
    logger.log_learner_start(step, target - step)
    while step < target:
        offset = ((step - start_step) % full_batches) * effective_batch
        batch = [tokenize(record, tokenizer, config)
                 for record in transitions[offset:offset + effective_batch]]
        loss = network.update(batch)  # OOMs propagate; only successful updates advance the step.
        step += 1
        storage.save_latest_checkpoint(step, network)
        save_progress(progress_path, {'completed_updates': step - start_step, 'solver_step': step, 'complete': False})
        if step % config.checkpoint_interval == 0 or step == target:
            storage.save_checkpoint(step, network)
        if step % config.log_interval == 0 or step == target:
            logger.log_training(step, loss, network.optimizer.param_groups[0]['lr'], None, len(transitions))
        if step % config.validation_interval == 0 and step < target:
            validation_loss = evaluate(network, validation, tokenizer, config)
            logger.wandb_run.log({'learner/step': step, 'sft_validation/loss': validation_loss})
            print(f'Step {step}: fixed SFT validation loss {validation_loss:.6f}', flush=True)
    validation_loss = evaluate(network, validation, tokenizer, config)
    logger.wandb_run.log({'learner/step': step, 'sft_validation/loss': validation_loss})
    print(f'Step {step}: fixed SFT validation loss {validation_loss:.6f}', flush=True)
    save_progress(progress_path, {'completed_updates': num_steps, 'solver_step': step, 'complete': True})
    print(f'Completed {args.round_dir.name} solver training at step {step}.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--round-dir', type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.run_dir = args.run_dir.resolve()
    args.round_dir = args.round_dir.resolve()
    experiment = load_experiment_config(args.config, args.run_dir.name)
    config = experiment.rl
    epochs = yaml.safe_load(args.config.read_text())['experiment']['deltaproof']['learner_epochs_per_round']
    settings = load_stp_wandb_settings(args.config, args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if has_run_config(args.run_dir):
        saved = load_run_config(args.run_dir)
        saved_config = rl_config_from_dict(saved['config'], args.run_dir.name)
        if changed_config_fields(saved_config, config) or saved['wandb_run_id'] != settings['id']:
            raise ValueError('Solver configuration or W&B ID differs from the saved run.')
    else:
        save_run_config(args.run_dir, config, settings['id'])
    logger = RunLogger(args.run_dir, config.reward_window, initialize_stp_wandb(settings, config))
    try:
        train_round(args, config, experiment.sft, epochs, logger)
    except BaseException as error:
        logger.log_crash(error)
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)


if __name__ == '__main__':
    main()
