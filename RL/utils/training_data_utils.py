"""Shared STP proof selection, independent of either training environment."""

PROVER_PROMPT = 'Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'

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


def select_training_proofs(generated_proofs, existing_examples=()):
    """Keep verified dataset proofs and conjectures with pass rate <= 0.5.

    Select the most recent copy of each formatted whole proof. Multiplicity
    contributes to the empirical pass rate, as in the STP LLM trainer.
    """
    results = {}
    for proof in generated_proofs:
        results.setdefault(proof['statement'], [0, 0])
        count = proof.get('multiplicity', 1)
        results[proof['statement']][0] += count * proof.get('complete', False)
        results[proof['statement']][1] += count
    known = {(example['prompt'], example['target']) for example in existing_examples}
    selected = []
    for proof in reversed(generated_proofs):
        if not proof.get('complete', False):
            continue
        successes, attempts = results[proof['statement']]
        if get_conjecture_level(proof) > 0 and successes / attempts > 0.5:
            continue
        key = format_data(proof.get('header'), proof['statement'], proof['proof'])
        if key not in known:
            known.add(key)
            selected.append(proof)
    return selected
