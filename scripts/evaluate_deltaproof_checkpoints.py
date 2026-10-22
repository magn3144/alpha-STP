"""Evaluate saved DeltaProof round snapshots without changing training runs."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from scripts.evaluate_llm_checkpoints import export_summary, write_json


def evaluation_plan(config):
    evaluations = []
    for label, run_dir in config['checkpoint_runs'].items():
        completed_games = 0
        for round_id in config['rounds']:
            round_dir = Path(run_dir) / f'round{round_id}'
            assert (round_dir / 'round_complete').is_file(), round_dir
            metrics = json.loads((round_dir / 'round_metrics.json').read_text())
            completed_games += metrics['attempts']
            evaluations.append({
                'run': label, 'round': round_id, 'game': completed_games,
                'seed': config['checkpoint_seed'],
                'checkpoint': str(round_dir / 'solver_checkpoint.pt'),
                'directory': f'{label}/round{round_id}',
            })
    for repeat, seed in enumerate(config['baseline_seeds']):
        evaluations.append({
            'run': 'parallel_sampling', 'repeat': repeat, 'game': 0,
            'seed': seed, 'checkpoint': config['starting_checkpoint'],
            'directory': f'parallel_sampling/repeat{repeat}',
        })
    for evaluation in evaluations:
        assert Path(evaluation['checkpoint']).stat().st_size > 0
    return evaluations


def summarize(evaluation, directory, theorems, overlaps):
    rows = [json.loads(line) for line in (directory / 'inference.jsonl').read_text().splitlines()]
    results = [row for row in rows if row['type'] == 'result']
    assert rows[-1]['type'] == 'summary'
    assert len(results) == len(theorems)
    assert {int(row['theorem_id']) for row in results} == set(range(len(theorems)))
    results.sort(key=lambda row: int(row['theorem_id']))
    solved = []
    for index, result in enumerate(results):
        assert result['request']['theorem'] == theorems[index]
        solved.append(result['status'] == 'proved')
    write_json(directory / 'theorem_results.json', [
        {
            'theorem_id': i, 'formal_statement': theorem,
            'overlaps_rl_training': i in overlaps, 'solved': solved[i],
            'proof': results[i]['proof'], 'error': results[i]['error'],
        }
        for i, theorem in enumerate(theorems)
    ])
    disjoint = [i for i in range(len(theorems)) if i not in overlaps]
    return evaluation | {
        'theorem_count': len(theorems), 'solved_theorems': sum(solved),
        'solve_rate': sum(solved) / len(theorems),
        'disjoint_theorem_count': len(disjoint),
        'disjoint_solved_theorems': sum(solved[i] for i in disjoint),
        'disjoint_solve_rate': sum(solved[i] for i in disjoint) / len(disjoint),
    }


def main(args):
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    source = Path(config['source_theorems'])
    theorems = json.loads(source.read_text())
    audit = json.loads(Path(config['overlap_audit']).read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit['validation_sha256']
    assert len(theorems) == config['theorem_count'] == len(set(theorems))
    assert len(set(config['baseline_seeds'])) == len(config['baseline_seeds']) > 1
    evaluations = evaluation_plan(config)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    lean_project = Path(config['lean_project'])
    saved_config = json.loads(Path(config['solver_config']).read_text())
    manifest = {
        'config': config, 'evaluations': evaluations, 'solver_config': saved_config,
        'validation_sha256': audit['validation_sha256'],
        'lean_toolchain': (lean_project / 'lean-toolchain').read_text().strip(),
        'lean_manifest_sha256': hashlib.sha256((lean_project / 'lake-manifest.json').read_bytes()).hexdigest(),
    }
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Evaluation inputs changed.'
    else:
        write_json(manifest_path, manifest)
    requests_path = output_dir / 'requests.jsonl'
    requests_path.write_text(''.join(json.dumps({
        'request_id': f'validation-{i}', 'theorem_id': i,
        'source': 'dataset', 'attempt': 0, 'theorem': theorem,
    }) + '\n' for i, theorem in enumerate(theorems)))
    print(f'{len(evaluations)} evaluations, {len(theorems)} theorems, '
          f'{saved_config["config"]["num_simulations"]} simulations per search.', flush=True)
    if args.prepare_only:
        print('Preparation complete. No GPU code imported or executed.', flush=True)
        return
    summaries = []
    for evaluation in evaluations:
        directory = output_dir / evaluation['directory']
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            # A private run view lets the existing inference loader read this snapshot.
            # The original run and its latest checkpoint are never modified.
            run_view = directory / 'model'
            (run_view / 'checkpoints').mkdir(parents=True, exist_ok=True)
            write_json(run_view / 'config_01.json', saved_config)
            checkpoint_link = (
                run_view / 'network_params.pt'
                if evaluation['run'] == 'parallel_sampling'
                else run_view / 'checkpoints/latest.pt'
            )
            if not checkpoint_link.is_symlink():
                checkpoint_link.symlink_to(evaluation['checkpoint'])
            assert checkpoint_link.resolve() == Path(evaluation['checkpoint']).resolve()
            search = saved_config['config']
            command = [
                config['python'], '-u', '-m', 'alphaproof.inference.infer',
                '--config', config['experiment_config'],
                '--input', str(requests_path), '--batch-id', evaluation['directory'],
                '--run-dir', str(run_view), '--lean-project', str(lean_project),
                '--import', 'Mathlib', '--seed', str(evaluation['seed']),
            ]
            for option in ['num_simulations', 'num_sampled_actions', 'tactic_timeout',
                           'final_check_timeout', 'max_concurrent_lean_imports',
                           'inference_num_gpus', 'inference_batch_size', 'inference_batch_timeout']:
                command.extend(['--' + option.replace('_', '-'), str(search[option])])
            command.extend(['--parallel-searches', str(search['num_actors'])])
            print(f'Evaluating {evaluation["directory"]}, seed={evaluation["seed"]}', flush=True)
            with (directory / 'inference.tmp').open('w') as protocol:
                subprocess.run(command, cwd=config['repo_dir'], stdout=protocol, check=True)
            (directory / 'inference.tmp').replace(directory / 'inference.jsonl')
            write_json(summary_path, summarize(evaluation, directory, theorems, set(audit['overlap_indices'])))
        summaries.append(json.loads(summary_path.read_text()))
        export_summary(output_dir, summaries, len(config['baseline_seeds']))
    (output_dir / 'SUCCESS').touch()
    print(f'Evaluation complete: {output_dir / "summary.csv"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())
