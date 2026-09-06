import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from utils.RL_utils import CONJECTURE_THRESHOLD, get_conjecture_level


def load_deltaproof_dataset(path, dataset_size):
    records = []
    with open(path, encoding='utf-8') as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue
            raw = json.loads(line)
            theorem = raw['theorem'].rstrip()
            assert theorem.endswith(':= by sorry')
            records.append({
                'statement': theorem.removesuffix('sorry').rstrip(),
                'label': ['deltaproof'],
                'header': None,
                'matching_weight': 1,
            })
    if dataset_size:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(records), dataset_size, replace=False)
        records = [records[index] for index in indices]
    return records


def select_dataset_theorems(dataset, solved, count, seed):
    unsolved = {
        item['lemma_id']: item
        for item in dataset
        if item['lemma_id'] not in solved
    }
    unsolved = list(unsolved.values())
    if not unsolved:
        return []
    if len(unsolved) < count:
        raise ValueError(
            f'Only {len(unsolved)} unsolved dataset theorems remain; '
            f'{count} are required for this round.'
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unsolved)
    return deepcopy(unsolved[:count])


def select_conjecture_inputs(sampler, count, seed):
    rng = np.random.default_rng(seed)
    possible_inputs = [
        test_info
        for test_info in reversed(sampler.generated_proofs)
        if test_info.get('complete', False)
        and sampler.succ_rates[test_info['lemma_id']] > CONJECTURE_THRESHOLD + 0.1
        and get_conjecture_level(test_info) == 0
    ]
    rng.shuffle(possible_inputs)
    deduplicated = set()
    lemma_count = {}
    limit = max(1, round(count * 0.1))
    inputs = []
    for test_info in possible_inputs:
        invoked_lemmas = list(test_info.get('invokes', []))
        if rng.random() < 0.5:
            invoked_lemmas.append('')
        for invoked_lemma in invoked_lemmas:
            if invoked_lemma not in sampler.avaliable_lemmas:
                continue
            key = (test_info['statement'], invoked_lemma)
            if key in deduplicated:
                continue
            if lemma_count.get(invoked_lemma, 0) >= limit:
                continue
            deduplicated.add(key)
            lemma_count[invoked_lemma] = lemma_count.get(invoked_lemma, 0) + 1
            inputs.append(test_info | {
                'shared_lemma': invoked_lemma,
                'shared_lemma_statement': sampler.avaliable_lemmas[invoked_lemma],
            })
            if len(inputs) == count:
                return inputs
    return inputs


def deduplicate_conjectures(candidates, dataset, known_statements):
    known = {
        (item.get('header'), normalize_statement(item['statement']))
        for item in dataset
    }
    known.update((None, normalize_statement(statement)) for statement in known_statements)
    deduplicated = []
    for candidate in candidates:
        key = (
            candidate.get('header'),
            normalize_statement(candidate['statement']),
        )
        if key in known:
            continue
        known.add(key)
        labels = [
            label
            for label in candidate['label']
            if not label.startswith('conjecture')
        ]
        labels.append('conjecture 1')
        candidate['label'] = labels
        deduplicated.append(candidate)
    return deduplicated


def normalize_statement(statement):
    return statement.replace(' ', '').replace('\n', '')


def build_requests(dataset_theorems, conjectures, conjecture_attempts, round_id):
    requests = []
    test_infos = {}
    for test_info in dataset_theorems:
        add_request(requests, test_infos, test_info, 'dataset', 0, round_id)
    for test_info in conjectures:
        for attempt in range(conjecture_attempts):
            add_request(
                requests,
                test_infos,
                test_info,
                'conjecture',
                attempt,
                round_id,
            )
    np.random.default_rng(round_id).shuffle(requests)
    return requests, test_infos


def add_request(requests, test_infos, test_info, source, attempt, round_id):
    theorem_id = f'{source}:{test_info["lemma_id"]}'
    request_id = f'round{round_id}:{theorem_id}:{attempt}'
    requests.append({
        'request_id': request_id,
        'theorem_id': theorem_id,
        'source': source,
        'attempt': attempt,
        'header': test_info.get('header'),
        'theorem': test_info['statement'] + ' sorry',
    })
    test_infos[request_id] = deepcopy(test_info)


def apply_results(results, test_infos):
    generated_proofs = []
    for result in results:
        test_info = test_infos[result['request_id']]
        test_info.update({
            'request_id': result['request_id'],
            'source': result['source'],
            'attempt': result['attempt'],
            'complete': result['status'] == 'proved',
            'pass': result['status'] == 'proved',
            'proof': result['proof'] or '',
            'system_messages': result['error'],
            'verify_time': result['duration_seconds'],
            'simulations_allocated': result['simulations_allocated'],
            'simulations_used': result['simulations_used'],
            'transition_count': result['transition_count'],
        })
        generated_proofs.append(test_info)
    return generated_proofs


def read_jsonl(path):
    with open(path, encoding='utf-8') as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output_file:
        for record in records:
            output_file.write(json.dumps(record) + '\n')
    temporary_path.replace(path)
