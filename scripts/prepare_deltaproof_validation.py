"""Prepare fixed validation transitions from Lean's original compiler traces."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from leantree import LeanProject
from leantree.repl_adapter.data import ReplGoalInfo


EXTRACTION = 'source-tactic-compiler-traces-v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_jsonl(path, records):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in records))
    temporary.replace(path)


def trace_transitions(edges, theorem_id):
    """Use recorded goals and tactics directly, without rewriting or replay."""
    by_id = {}
    for step in edges:
        goal = step['goalBefore']['id']
        if goal in by_id:
            assert by_id[goal] == step, theorem_id
        by_id[goal] = step
    depths = {}
    visiting = set()

    def depth(goal):
        if goal in depths:
            return depths[goal]
        assert goal not in visiting, theorem_id
        visiting.add(goal)
        children = by_id[goal]['children']
        assert all(child in by_id for child in children), theorem_id
        depths[goal] = 1 + max([0] + [depth(child) for child in children])
        visiting.remove(goal)
        return depths[goal]

    for goal in by_id:
        depth(goal)
    records = []
    for step in by_id.values():
        action = step['tacticString'].strip()
        assert action and 'sorry' not in action and 'admit' not in action, theorem_id
        state = str(ReplGoalInfo.goal_from_repl_data(step['goalBefore']))
        records.append(dict(id=theorem_id, state=state,
                            action=action, value=-float(depths[step['goalBefore']['id']])))
    return records


def extract_batch(arguments):
    dataset, start, project_path, output = arguments
    project = LeanProject(project_path)
    helper = Path(__file__).with_name('extract_deltaproof_traces.lean').resolve()
    by_index = {}
    pending = []
    for index, row in enumerate(dataset, start):
        cache = output / 'compiler_traces' / f'{index:03d}.json'
        identity = dict(extraction=EXTRACTION, extractor_sha256=sha256(helper), id=row['id'],
                        proof_sha256=hashlib.sha256(row['verified_proof'].encode()).hexdigest(),
                        lean_toolchain=project.lean_toolchain())
        if cache.exists():
            cached = json.loads(cache.read_text())
            assert cached['identity'] == identity, cache
            by_index[index] = cached['records']
        else:
            pending.append((index, row, cache, identity))
    if not pending:
        return [record for index in range(start, start + len(dataset)) for record in by_index[index]]
    for index, row, cache, identity in pending:
        source = output / 'compiler_sources' / f'{index:03d}.lean'
        source.write_text(row['verified_proof'])
        imports = [line.strip() for line in row['verified_proof'].splitlines() if line.startswith('import ')]
        assert imports == ['import Mathlib'], row['id']
    request = output / 'compiler_sources' / f'request{start:03d}.json'
    write_json(request, [dict(index=index, path=str((output / 'compiler_sources' / f'{index:03d}.lean').resolve()))
                         for index, _, _, _ in pending])
    environment = os.environ.copy()
    library = project.repl_path.parent.parent / 'lib/lean'
    environment['LEAN_PATH'] = str(library) + ':' + environment.get('LEAN_PATH', '')
    command = ['lake', 'env', 'lean', '--run', str(helper), str(request.resolve())]
    with (output / 'compiler_sources' / f'errors{start:03d}.log').open('w') as errors:
        with subprocess.Popen(command, cwd=project.path, env=environment,
                              stdout=subprocess.PIPE, stderr=errors, text=True) as process:
            expected = {index: (row, cache, identity) for index, row, cache, identity in pending}
            for line in process.stdout:
                result = json.loads(line)
                index = result['index']
                row, cache, identity = expected.pop(index)
                transitions = trace_transitions(result['steps'], row['id'])
                assert transitions, row['id']
                write_json(cache, dict(identity=identity, records=transitions))
                by_index[index] = transitions
                print(f'Compiled theorem {index + 1}/400; {len(transitions)} transitions', flush=True)
            assert process.wait() == 0, f'Compiler failed in batch {start}; see {errors.name}'
            assert not expected, expected.keys()
    return [record for index in range(start, start + len(dataset)) for record in by_index[index]]


def extract(dataset, project_path, output, batch_size, workers):
    """Compile isolated proofs; cache every completed theorem atomically."""
    for name in ['compiler_sources', 'compiler_traces']:
        (output / name).mkdir(parents=True, exist_ok=True)
    batches = [(dataset[start:start + batch_size], start, project_path, output)
               for start in range(0, len(dataset), batch_size)]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(extract_batch, batches):
            records.extend(batch)
    assert {r['id'] for r in records} == {r['id'] for r in dataset}
    return records


def load_prepared(dataset, output, validation_sha256):
    """Reject missing, incomplete, modified, or differently sourced inputs."""
    path = output / 'transitions.jsonl'
    assert (output / 'prepared.json').is_file(), 'Run scripts.prepare_deltaproof_validation on CPU first.'
    prepared = json.loads((output / 'prepared.json').read_text())
    assert prepared['extraction'] == EXTRACTION
    assert prepared['extractor_sha256'] == sha256(Path(__file__).with_name('extract_deltaproof_traces.lean'))
    assert prepared['validation_sha256'] == validation_sha256
    assert prepared['transitions_sha256'] == sha256(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert prepared['theorem_count'] == len(dataset) == 400
    assert prepared['transition_count'] == len(records)
    assert {r['id'] for r in records} == {r['id'] for r in dataset}
    assert all(r['state'] and r['action'] and math.isfinite(r['value']) and r['value'] <= -1 for r in records)
    return records, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    source = Path(config['validation_data'])
    dataset = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(dataset) == len({r['id'] for r in dataset}) == 400
    output = Path(config['output_dir'])
    records = extract(dataset, config['lean_project'], output,
                      config['extraction_batch_size'], config['extraction_workers'])
    path = output / 'transitions.jsonl'
    write_jsonl(path, records)
    write_json(output / 'prepared.json', dict(extraction=EXTRACTION,
               extractor_sha256=sha256(Path(__file__).with_name('extract_deltaproof_traces.lean')),
               validation_sha256=sha256(source),
               transitions_sha256=sha256(path), theorem_count=400, transition_count=len(records)))
    load_prepared(dataset, output, sha256(source))
    print(f'PREPARED: all 400 theorems; {len(records)} transitions', flush=True)


if __name__ == '__main__':
    main()
