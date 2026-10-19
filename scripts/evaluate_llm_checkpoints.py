"""Evaluate saved LLM rounds using STP generation and delta-proof's Lean environment."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import statistics
from pathlib import Path

import yaml

from RL.utils.experiment_utils import RL_DIR, run_python


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def evaluation_plan(config):
    evaluations = []
    for label, run_dir in config['checkpoint_runs'].items():
        completed_games = 0
        for round_id in config['rounds']:
            round_dir = Path(run_dir) / f'round{round_id}'
            assert (round_dir / 'round_complete').is_file(), round_dir
            records = json.loads((round_dir / 'actor_metrics.json').read_text())
            completed_games += sum(record['games'] for record in records.values())
            evaluations.append({
                'run': label, 'round': round_id, 'game': completed_games,
                'seed': config['checkpoint_seed'],
                'model': str(round_dir / 'RL_model'),
                'directory': f'{label}/round{round_id}',
            })
    for repeat, seed in enumerate(config['baseline_seeds']):
        evaluations.append({
            'run': 'parallel_sampling', 'repeat': repeat, 'game': 0,
            'seed': seed, 'model': config['starting_model'],
            'directory': f'parallel_sampling/repeat{repeat}',
        })
    for evaluation in evaluations:
        model = Path(evaluation['model'])
        assert (model / 'config.json').is_file(), model
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        for shard in set(index['weight_map'].values()):
            assert (model / shard).is_file() and (model / shard).stat().st_size > 0, model / shard
    return evaluations


def summarize(evaluation, output_dir, dataset, overlaps, samples_per_theorem):
    attempts = [0] * len(dataset)
    successes = [0] * len(dataset)
    with gzip.open(output_dir / 'generated_proofs_validation.jsonl.gz', 'rt') as stream:
        for line in stream:
            proof = json.loads(line)
            theorem_id = proof['lemma_id']
            assert proof['statement'] == dataset[theorem_id]['formal_statement'].rsplit('sorry', 1)[0].strip()
            attempts[theorem_id] += proof['multiplicity']
            successes[theorem_id] += proof['multiplicity'] * proof['complete']
    assert all(count == samples_per_theorem for count in attempts), attempts
    disjoint = [i for i in range(len(dataset)) if i not in overlaps]
    theorems = [
        {
            'theorem_id': i, 'formal_statement': row['formal_statement'],
            'overlaps_rl_training': i in overlaps,
            'attempts': attempts[i], 'successful_attempts': successes[i],
            'solved': successes[i] > 0,
        }
        for i, row in enumerate(dataset)
    ]
    write_json(output_dir / 'theorem_results.json', theorems)
    return evaluation | {
        'theorem_count': len(dataset), 'samples_per_theorem': samples_per_theorem,
        'solved_theorems': sum(count > 0 for count in successes),
        'solve_rate': sum(count > 0 for count in successes) / len(dataset),
        'sample_pass_rate': sum(successes) / sum(attempts),
        'disjoint_theorem_count': len(disjoint),
        'disjoint_solved_theorems': sum(successes[i] > 0 for i in disjoint),
        'disjoint_solve_rate': sum(successes[i] > 0 for i in disjoint) / len(disjoint),
    }


def export_summary(output_dir, summaries, baseline_repeats):
    fields = list(dict.fromkeys(key for summary in summaries for key in summary))
    with (output_dir / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    baseline = [summary for summary in summaries if summary['run'] == 'parallel_sampling']
    if len(baseline) == baseline_repeats:
        uncertainty = {'repeats': len(baseline), 'seeds': [row['seed'] for row in baseline]}
        for metric in ['solve_rate', 'disjoint_solve_rate']:
            rates = [row[metric] for row in baseline]
            deviation = statistics.stdev(rates)
            uncertainty[metric] = {
                'mean': statistics.mean(rates), 'standard_deviation': deviation,
                'standard_error': deviation / len(rates) ** 0.5,
                'minimum': min(rates), 'maximum': max(rates),
            }
        uncertainty['interpretation'] = 'Sampling variability across seeds on the same fixed theorem set; not uncertainty from sampling different theorems.'
        write_json(output_dir / 'starting_model_uncertainty.json', uncertainty)


def main(args):
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    dataset_path = Path(config['validation_data']).resolve()
    dataset = json.loads(dataset_path.read_text())
    source_path = Path(config['source_theorems'])
    source = json.loads(source_path.read_text())
    audit = json.loads(Path(config['overlap_audit']).read_text())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == audit['validation_sha256']
    assert len(dataset) == config['theorem_count'] == len(source)
    assert [row['formal_statement'] for row in dataset] == source
    assert len(set(source)) == len(source)
    assert len(set(config['baseline_seeds'])) == len(config['baseline_seeds'])
    assert len(config['baseline_seeds']) > 1
    evaluations = evaluation_plan(config)
    verification = config['verification']
    lean_project = Path(verification['lean_project'])
    assert os.access(verification['lake_path'], os.X_OK), verification['lake_path']
    output_dir = Path(config['output_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config': config, 'evaluations': evaluations,
        'validation_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'overlap_audit_sha256': hashlib.sha256(Path(config['overlap_audit']).read_bytes()).hexdigest(),
        'lean_toolchain': (lean_project / 'lean-toolchain').read_text().strip(),
        'lean_manifest_sha256': hashlib.sha256((lean_project / 'lake-manifest.json').read_bytes()).hexdigest(),
        'lean_header': 'import Mathlib\n',
    }
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Evaluation inputs changed.'
    else:
        write_json(manifest_path, manifest)
    dataset_config = output_dir / 'dataset_config.json'
    write_json(dataset_config, [{'dataset_path': str(dataset_path), 'weight': config['samples_per_theorem']}])
    total = len(evaluations) * len(dataset) * config['samples_per_theorem']
    print(f'{len(evaluations)} evaluations, {len(dataset)} theorems, pass@{config["samples_per_theorem"]}: {total:,} attempts.', flush=True)
    print(f'Also reporting the {len(dataset) - len(audit["overlap_indices"])}-theorem subset without detected training overlap.', flush=True)
    if args.prepare_only:
        print('Preparation complete. No GPU code imported or executed.', flush=True)
        return
    summaries = []
    for evaluation in evaluations:
        evaluation_dir = output_dir / evaluation['directory']
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        summary_path = evaluation_dir / 'summary.json'
        if not summary_path.exists():
            print(f'Evaluating {evaluation["directory"]}, seed={evaluation["seed"]}', flush=True)
            run_python(
                RL_DIR / 'generate_and_test.py',
                '--model', evaluation['model'], '--exp_dir', evaluation_dir,
                '--raw_dataset_config', dataset_config, '--save_file_name', 'validation',
                '--nr_samples', config['samples_per_theorem'],
                '--temperature', config['temperature'], '--seed', evaluation['seed'],
                '--lean-project', verification['lean_project'],
                '--lake-path', verification['lake_path'],
                '--lean-workers', verification['workers'],
                '--lean-timeout', verification['timeout'],
            )
            summary = summarize(evaluation, evaluation_dir, dataset, set(audit['overlap_indices']), config['samples_per_theorem'])
            write_json(summary_path, summary)
        summaries.append(json.loads(summary_path.read_text()))
        export_summary(output_dir, summaries, len(config['baseline_seeds']))
    print(f'Evaluation complete: {output_dir / "summary.csv"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Validate inputs and write the plan without importing GPU code.')
    main(parser.parse_args())
