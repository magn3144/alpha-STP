import argparse
from pathlib import Path

from utils.config_utils import load_experiment_config
from utils.experiment_utils import RL_DIR, run_external_python, run_python
from utils.timing_utils import configure_timing, timer


def latest_conjecturer(experiment, exp_dir, round_id):
    for previous_round in range(round_id - 1, -1, -1):
        checkpoint = exp_dir / f'round{previous_round}' / 'conjecturer_model'
        if checkpoint.is_dir():
            return checkpoint
    return experiment['conjecturer_model']


def run_llm_round(
    args,
    config_path,
    experiment,
    dataset_size,
    round_id,
    round_dir,
):
    if round_id == 0:
        model = experiment['base_model']
    else:
        model = Path(experiment['exp_dir']) / f'round{round_id - 1}' / 'RL_model'
    samples_key = 'first_round' if round_id == 0 else 'later_rounds'
    samples_per_statement = experiment['samples_per_statement'][samples_key]
    with timer('generation_step', round=round_id):
        run_python(
            RL_DIR / 'RL_step1_generate.py',
            '--model', model,
            '--exp_dir', round_dir,
            '--seed', round_id,
            '--temperature', experiment['temperature'],
            '--dataset_config', experiment['dataset_config'],
            '--sampler', experiment['sampler'],
            '--conjecture_multiplier', experiment['conjecture_multiplier'],
            '--samples_per_statement', samples_per_statement,
            '--dataset_size', dataset_size,
            dry_run=args.dry_run,
        )
    with timer('round_training_step', round=round_id):
        run_python(
            RL_DIR / 'RL_step2_train.py',
            '--base_model', model,
            '--exp_dir', round_dir,
            '--epoch', experiment['epochs'],
            '--training_config', config_path,
            dry_run=args.dry_run,
        )


def run_deltaproof_round(
    args,
    config_path,
    experiment,
    exp_dir,
    round_id,
    round_dir,
):
    solver = experiment['solver']
    model = latest_conjecturer(experiment, exp_dir, round_id)
    with timer('generation_step', round=round_id):
        run_python(
            RL_DIR / 'RL_step1_generate_deltaproof.py',
            '--config', config_path,
            '--model', model,
            '--exp_dir', round_dir,
            '--seed', round_id,
            dry_run=args.dry_run,
        )
    if not args.dry_run and (round_dir / 'experiment_complete').is_file():
        return False
    transitions_path = round_dir / 'deltaproof_transitions.jsonl'
    with timer('deltaproof_training_step', round=round_id):
        run_external_python(
            solver['python'],
            'alphaproof.training.train_transitions',
            '--config', solver['config'],
            '--run-dir', solver['run_dir'],
            '--input', transitions_path,
            '--batch-id', f'round{round_id}',
            '--num-steps', solver['learner_steps_per_round'],
            cwd=solver['repo_dir'],
            dry_run=args.dry_run,
        )
    if round_id > 0 and solver['conjecture_fraction']:
        with timer('conjecturer_training_step', round=round_id):
            run_python(
                RL_DIR / 'RL_step2_train.py',
                '--base_model', model,
                '--exp_dir', round_dir,
                '--epoch', experiment['epochs'],
                '--training_config', config_path,
                '--sft_dataset', experiment['conjecturer_sft_dataset'],
                '--conjecturer_only',
                '--model_name', 'conjecturer_model',
                dry_run=args.dry_run,
            )
    return True


def main(args):
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path, 'rl')
    experiment = config['experiment']
    training = config['training']
    if not 0 <= args.start_round < experiment['total_rounds']:
        raise ValueError('start_round must be between zero and total_rounds - 1')
    exp_dir = Path(experiment['exp_dir'])
    dataset_size = experiment['dataset_size']
    batch_size = training['trainer']['train_batch_size']
    configure_timing(exp_dir, new_session=True)
    print(
        f'Configuration: solver={experiment["solver"]["type"]}, '
        f'dataset_size={dataset_size}, batch_size={batch_size}',
        flush=True,
    )
    with timer('stp_run'):
        for round_id in range(args.start_round, experiment['total_rounds']):
            round_dir = exp_dir / f'round{round_id}'
            print(f'Starting self-play round {round_id}', flush=True)
            with timer('round', round=round_id):
                if experiment['solver']['type'] == 'llm':
                    run_llm_round(
                        args,
                        config_path,
                        experiment,
                        dataset_size,
                        round_id,
                        round_dir,
                    )
                else:
                    should_continue = run_deltaproof_round(
                        args,
                        config_path,
                        experiment,
                        exp_dir,
                        round_id,
                        round_dir,
                    )
                    if not should_continue:
                        print('All dataset theorems are solved.', flush=True)
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sequential STP self-play rounds.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
