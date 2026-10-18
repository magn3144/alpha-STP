"""Shared STP dataset sampling and proof selection."""

from copy import deepcopy

PROVER_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'


def select_dataset_theorems(dataset, solved, count, rng, attempts_per_theorem=1):
    """Fill the attempt budget with shuffled passes over unsolved theorem groups."""
    unsolved = list({
        item['lemma_id']: item
        for item in dataset
        if item['lemma_id'] not in solved
    }.values())
    if not unsolved:
        return []
    assert count % attempts_per_theorem == 0
    theorem_count = count // attempts_per_theorem
    selected = []
    while len(selected) < theorem_count:
        rng.shuffle(unsolved)
        selected.extend(unsolved[:theorem_count - len(selected)])
    return [deepcopy(item) for item in selected for _ in range(attempts_per_theorem)]


def get_conjecture_level(test_info):
    for label in test_info['label']:
        if label.startswith('conjecture '):
            return int(label.split()[-1])
    return 0


def format_data(header, statement, proof):
    if header is not None:
        prompt = 'Complete the following Lean 4 code:\n\n```lean4\n' + header + statement.strip()
    else:
        prompt = f'{PROVER_PROMPT}\n{statement.strip()}'
    return prompt, proof


def select_training_proofs(generated_proofs, existing_examples=(), filter_dataset=False):
    """Keep verified proofs, filtering conjectures and optionally dataset theorems.

    Select the most recent copy of each formatted whole proof. Multiplicity
    contributes to the empirical pass rate. Dataset filtering uses each round's
    attempts separately so replay does not change its measured solve rate.
    """
    results = {}
    for proof in generated_proofs:
        key = (proof['round'] if filter_dataset and get_conjecture_level(proof) == 0 else None,
               proof['statement'])
        results.setdefault(key, [0, 0])
        count = proof.get('multiplicity', 1)
        results[key][0] += count * proof.get('complete', False)
        results[key][1] += count
    known = {(example['prompt'], example['target']) for example in existing_examples}
    selected = []
    for proof in reversed(generated_proofs):
        if not proof.get('complete', False):
            continue
        key = (proof['round'] if filter_dataset and get_conjecture_level(proof) == 0 else None,
               proof['statement'])
        successes, attempts = results[key]
        if (filter_dataset or get_conjecture_level(proof) > 0) and successes / attempts > 0.5:
            continue
        key = format_data(proof.get('header'), proof['statement'], proof['proof'])
        if key not in known:
            known.add(key)
            selected.append(proof)
    return selected
