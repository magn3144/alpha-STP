import argparse
from pathlib import Path

from utils.experiment_utils import RL_DIR, run_python


def main(args):
    for round_id in range(args.start_round, args.total_rounds):
        round_dir = Path(args.exp_dir) / f'round{round_id}'
        print(f'Starting sampling round {round_id} with model {args.base_model}', flush=True)
        run_python(
            RL_DIR / 'RL_step1_generate.py',
            '--model', args.base_model,
            '--exp_dir', round_dir,
            '--seed', round_id,
            '--temperature', args.temperature,
            '--dataset_config', args.dataset_config,
            '--sampler', 'Sampler_naive',
            '--samples_per_statement', args.samples_per_statement,
            '--statements_per_round', args.statements_per_round,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample proofs independently across rounds.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dataset-config', default=RL_DIR / 'dataset_configs/leanworkbook.json')
    parser.add_argument('--start-round', type=int, default=0)
    parser.add_argument('--total-rounds', type=int, default=8)
    parser.add_argument('--samples-per-statement', type=int, default=64)
    parser.add_argument('--statements-per-round', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
