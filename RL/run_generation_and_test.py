import argparse

from utils.experiment_utils import RL_DIR, run_python


def main(args):
    run_python(
        RL_DIR / 'generate_and_test.py',
        '--model', args.model,
        '--exp_dir', args.exp_dir,
        '--temperature', args.temperature,
        '--save_file_name', args.save_file_name,
        '--raw_dataset_config', args.dataset_config,
        '--seed', args.seed,
        '--nr_samples', args.nr_samples,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and verify benchmark proofs.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--dataset-config', default=RL_DIR / 'dataset_configs/miniF2F_ProofNet.json')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--save-file-name', default='tests')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nr-samples', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
