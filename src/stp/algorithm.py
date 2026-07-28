import hashlib
import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence, TypeVar

import numpy as np

from stp.records import (
    Conjecture,
    ConjectureAssessment,
    ConjectureFilterMetric,
    ConjectureInput,
    ProofRequest,
    SolveAttempt,
    Statement,
    TrainingExample,
)

Candidate = TypeVar("Candidate")


START_LEMMA_STMT = "<easy theorem>"
START_THM = "<hard theorem>"
END_THM = "</hard theorem>"
INVOKED_LEMMA = "<lemma>"
TRIVIAL_LEMMA = "theorem true : True"
CONJECTURE_SOURCE_THRESHOLD = 0.35
LEAN_CODE_PROMPT = "Complete the following Lean 4 code:\n\n```lean4\n"
PROVER_PROMPT = (
    LEAN_CODE_PROMPT
    + "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n"
)


def prover_prompt(statement: str, header: str | None) -> str:
    """Build the original STP whole-proof prompt."""

    if header is not None:
        return LEAN_CODE_PROMPT + header + statement.strip()
    return f"{PROVER_PROMPT}\n{statement.strip()}"


def _conjecturer_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the shared paper-style conjecturer prompt prefix."""

    easy_theorem = seed_statement + seed_proof
    return (
        LEAN_CODE_PROMPT
        + f"{INVOKED_LEMMA}\n{shared_lemma_statement.strip()}\n"
        + f"{START_LEMMA_STMT}\n{easy_theorem.strip()}\n"
        + START_THM
    )


def conjecturer_generation_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the conjecturer prompt used for autoregressive generation."""

    return (
        _conjecturer_prompt(
            shared_lemma_statement,
            seed_statement,
            seed_proof,
        )
        + "\n theorem"
    )


def conjecturer_training_prompt(
    shared_lemma_statement: str,
    seed_statement: str,
    seed_proof: str,
) -> str:
    """Build the same conjecturer prefix used for target-only training."""

    return _conjecturer_prompt(
        shared_lemma_statement,
        seed_statement,
        seed_proof,
    )


def parse_proof(text: str) -> str:
    """Extract a tactic proof from a model completion."""

    return text.split("\n```", 1)[0]


def parse_conjecture(text: str) -> str:
    """Extract a theorem declaration and append the empty `by` proof."""

    statement = "theorem " + text.split(END_THM, 1)[0].strip()
    if ":=" in statement:
        statement = statement.split(":=", 1)[0]
    return statement + ":= by"


def make_requests(
    statements: Sequence[Statement | Conjecture],
    attempts: int,
    seed: int,
) -> list[ProofRequest]:
    """Expand statements into deterministically seeded proof requests."""

    requests = []
    for statement_index, statement in enumerate(statements):
        source = "conjecture" if isinstance(statement, Conjecture) else "dataset"
        for attempt in range(attempts):
            request_seed = seed + statement_index * attempts + attempt
            requests.append(
                ProofRequest(
                    id=f"{statement.id}:{attempt}",
                    statement_id=statement.id,
                    statement=statement.statement,
                    header=statement.header,
                    attempt=attempt,
                    seed=request_seed,
                    source=source,
                )
            )
    return requests


def deduplicate_attempts(attempts: Sequence[SolveAttempt]) -> list[SolveAttempt]:
    """Collapse identical solver outputs while preserving multiplicity."""

    positions: dict[tuple[str, str | None, str], int] = {}
    result: list[SolveAttempt] = []
    for attempt in attempts:
        key = (attempt.statement_id, attempt.proof, attempt.status)
        if key in positions:
            index = positions[key]
            result[index] = replace(
                result[index],
                duration_seconds=(
                    result[index].duration_seconds + attempt.duration_seconds
                ),
                generated_tokens=(
                    result[index].generated_tokens + attempt.generated_tokens
                ),
                multiplicity=result[index].multiplicity + attempt.multiplicity,
            )
        else:
            positions[key] = len(result)
            result.append(attempt)
    return result


def screen_conjectures(conjectures: Sequence[Conjecture]) -> list[Conjecture]:
    """Baseline screening hook; currently retain every conjecture."""

    return list(conjectures)


def rank_conjectures(conjectures: Sequence[Conjecture]) -> list[Conjecture]:
    """Baseline value-head hook; currently preserve generation order."""

    return list(conjectures)


def assess_conjectures(
    attempts: Sequence[SolveAttempt],
) -> list[ConjectureAssessment]:
    """Compute original STP solve-rate difficulty assessments."""

    grouped: dict[str, list[SolveAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.statement_id].append(attempt)

    assessments = []
    for statement_id, group in grouped.items():
        total = sum(item.multiplicity for item in group)
        solved = sum(
            item.multiplicity for item in group if item.status == "proved"
        )
        assessments.append(
            ConjectureAssessment(
                statement_id=statement_id,
                method="solve_rate",
                score=solved / total,
                attempts=total,
                successes=solved,
                metrics={},
            )
        )
    return assessments


def select_dataset_statements(
    statements: Sequence[Statement],
    previous_assessments: Sequence[ConjectureAssessment],
    limit: int,
    seed: int,
) -> list[Statement]:
    """Select unsolved data plus a small replay of solved statements."""

    rates = {item.statement_id: item.score for item in previous_assessments}
    rng = random.Random(seed)
    selected = [
        statement
        for statement in statements
        if rates.get(statement.id, 0.0) == 0.0
        or rates.get(statement.id, 0.0) < 0.35
        or rng.random() < 0.05
    ]
    rng.shuffle(selected)
    return selected if limit <= 0 else selected[:limit]


def prepare_conjecture_inputs(
    statements: Sequence[Statement],
    attempts: Sequence[SolveAttempt],
    assessments: Sequence[ConjectureAssessment],
    declarations: dict[str, str],
    unproved_count: int,
    trivial_lemma_probability: float,
    lemma_cap_fraction: float,
    seed: int,
) -> list[ConjectureInput]:
    """Prepare paper-style theorem/proof/lemma conjecturer inputs."""

    by_id = {statement.id: statement for statement in statements}
    rates = {assessment.statement_id: assessment.score for assessment in assessments}
    if unproved_count == 0:
        return []

    proved = []
    seen_proofs = set()
    for attempt in attempts:
        key = (attempt.statement_id, attempt.proof)
        if (
            attempt.status == "proved"
            and attempt.proof is not None
            and attempt.statement_id in by_id
            and rates.get(attempt.statement_id, 0.0)
            > CONJECTURE_SOURCE_THRESHOLD
            and key not in seen_proofs
        ):
            seen_proofs.add(key)
            proved.append(attempt)

    rng = random.Random(seed)
    candidates = []
    for attempt in proved:
        lemma_names = [
            name for name in attempt.invoked_lemmas if name in declarations
        ]
        if rng.random() < trivial_lemma_probability:
            lemma_names.append("")
        for name in lemma_names:
            candidates.append((attempt, name))
    rng.shuffle(candidates)

    cap = math.ceil(lemma_cap_fraction * unproved_count)
    lemma_counts: dict[str, int] = defaultdict(int)
    used_pairs = set()
    retained = []
    for attempt, name in candidates:
        pair = (attempt.statement_id, name)
        if pair in used_pairs or lemma_counts[name] >= cap:
            continue
        used_pairs.add(pair)
        lemma_counts[name] += 1
        retained.append((attempt, name))

    weighted = []
    for attempt, name in retained:
        statement = by_id[attempt.statement_id]
        priority = rng.random() ** (1.0 / statement.matching_weight)
        weighted.append((priority, attempt, name))
    weighted.sort(key=lambda item: item[0], reverse=True)

    inputs = []
    for index, (_, attempt, name) in enumerate(weighted[:unproved_count]):
        statement = by_id[attempt.statement_id]
        identity = f"{attempt.request_id}\n{name}"
        input_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        inputs.append(
            ConjectureInput(
                id=input_id,
                seed_statement_id=statement.id,
                seed_statement=statement.statement,
                seed_proof=attempt.proof or "",
                source_attempt_id=attempt.request_id,
                shared_lemma=name,
                shared_lemma_statement=(
                    TRIVIAL_LEMMA if name == "" else declarations[name]
                ),
                trivial_lemma=name == "",
                header=statement.header,
                labels=statement.labels,
                matching_weight=statement.matching_weight,
                generation_seed=seed + index,
            )
        )
    return inputs


def wasserstein_matching(
    candidates: Sequence[Candidate],
    candidate_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_weights: Sequence[float],
    capacity: float = 3.0,
) -> list[tuple[Candidate, float]]:
    """Greedily match conjectures to target embeddings as in original STP."""

    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    similarities = (
        candidate_embeddings / candidate_norms
    ) @ (target_embeddings / target_norms).T
    weights = np.zeros(len(candidates), dtype=np.float64)
    active = np.ones(len(candidates), dtype=bool)
    total_weight = float(sum(target_weights))
    for target_index, target_weight in enumerate(target_weights):
        count = min(int(target_weight), int(active.sum()))
        if count == 0:
            continue
        scores = np.where(active, similarities[:, target_index], -np.inf)
        chosen = np.argpartition(scores, -count)[-count:]
        for candidate_index in chosen:
            weights[candidate_index] += len(candidates) / total_weight
            if weights[candidate_index] > capacity:
                active[candidate_index] = False
    return [
        (candidate, float(weight))
        for candidate, weight in zip(candidates, weights, strict=True)
        if weight > 0
    ]


def build_training_examples(
    statements: Sequence[Statement | Conjecture],
    attempts: Sequence[SolveAttempt],
    assessments: Sequence[ConjectureAssessment],
    round_index: int,
    proof_threshold: float,
) -> list[TrainingExample]:
    """Build weighted prover examples from verified results."""

    by_id = {statement.id: statement for statement in statements}
    rates = {assessment.statement_id: assessment.score for assessment in assessments}
    success_counts: dict[str, int] = defaultdict(int)
    for attempt in attempts:
        if attempt.status == "proved":
            success_counts[attempt.statement_id] += attempt.multiplicity

    examples = []
    seen = set()
    for attempt in attempts:
        statement = by_id.get(attempt.statement_id)
        if (
            statement is None
            or attempt.status != "proved"
            or attempt.proof is None
            or rates[attempt.statement_id] > proof_threshold
        ):
            continue
        key = (attempt.statement_id, attempt.proof)
        if key in seen:
            continue
        seen.add(key)
        weight = math.exp(
            -0.001 * len(attempt.proof) - 0.01 * attempt.verify_seconds
        ) / success_counts[attempt.statement_id]
        examples.append(
            TrainingExample(
                prompt=prover_prompt(statement.statement, statement.header),
                target=attempt.proof,
                weight=weight,
                kind="proof",
                statement_id=statement.id,
                round=round_index,
            )
        )
    return examples


def select_conjecture_training_examples(
    conjectures: Sequence[Conjecture],
    attempts: Sequence[SolveAttempt],
    assessments: Sequence[ConjectureAssessment],
    round_index: int,
    pass_rate_threshold: float,
    elegance_drop_fraction: float,
    unfocused_ratio: int,
    unfocused_minimum: int,
    seed: int,
) -> tuple[list[TrainingExample], list[ConjectureFilterMetric]]:
    """Filter low-pass-rate conjectures and build unweighted examples."""

    rates = {
        assessment.statement_id: assessment.score
        for assessment in assessments
    }
    grouped: dict[str, list[SolveAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.statement_id].append(attempt)

    named = []
    trivial = []
    metrics: dict[str, ConjectureFilterMetric] = {}
    for conjecture in conjectures:
        successful = [
            attempt
            for attempt in grouped.get(conjecture.id, [])
            if attempt.status == "proved" and attempt.proof is not None
        ]
        successes = sum(attempt.multiplicity for attempt in successful)
        rate = rates.get(conjecture.id, 0.0)
        reused = any(
            conjecture.shared_lemma in attempt.invoked_lemmas
            for attempt in successful
        )
        minimum_length = (
            min(len(attempt.proof or "") for attempt in successful)
            if successful
            else None
        )
        if not 0.0 < rate <= pass_rate_threshold:
            reason = "pass_rate"
        elif not conjecture.shared_lemma and successful:
            reason = None
            trivial.append(conjecture)
        elif successful and reused:
            reason = None
            named.append(conjecture)
        else:
            reason = "lemma_not_reused"
        metrics[conjecture.id] = ConjectureFilterMetric(
            statement_id=conjecture.id,
            pass_rate=rate,
            successes=successes,
            trivial_lemma=conjecture.shared_lemma == "",
            named_lemma_reused=reused,
            minimum_proof_length=minimum_length,
            elegance_score=(
                minimum_length / len(conjecture.statement)
                if minimum_length is not None
                else None
            ),
            selected=False,
            rejection_reason=reason,
        )

    rng = random.Random(seed)
    rng.shuffle(named)
    rng.shuffle(trivial)
    trivial_limit = max(
        unfocused_ratio * len(named),
        unfocused_minimum,
    )
    for conjecture in trivial[trivial_limit:]:
        metric = metrics[conjecture.id]
        metrics[conjecture.id] = replace(
            metric,
            rejection_reason="unfocused_cap",
        )
    candidates = [*named, *trivial[:trivial_limit]]

    scores = []
    for conjecture in candidates:
        score = metrics[conjecture.id].elegance_score
        assert score is not None
        scores.append(score)
    scores.sort()
    cutoff = (
        scores[min(int(len(scores) * elegance_drop_fraction), len(scores) - 1)]
        if scores
        else math.inf
    )
    selected = []
    for conjecture in candidates:
        metric = metrics[conjecture.id]
        assert metric.elegance_score is not None
        if metric.elegance_score < cutoff:
            metrics[conjecture.id] = replace(
                metric,
                rejection_reason="elegance",
            )
            continue
        metrics[conjecture.id] = replace(
            metric,
            selected=True,
            rejection_reason=None,
        )
        target = (
            f"\n{conjecture.statement.removesuffix(':= by').strip()}\n"
            f"{END_THM}"
        )
        selected.append(
            TrainingExample(
                prompt=conjecturer_training_prompt(
                    conjecture.shared_lemma_statement,
                    conjecture.easy_statement,
                    conjecture.easy_proof,
                ),
                target=target,
                weight=1.0,
                kind="conjecture",
                statement_id=conjecture.id,
                round=round_index,
            )
        )
    ordered_metrics = [
        metrics[conjecture.id]
        for conjecture in conjectures
    ]
    return selected, ordered_metrics


def deduplicate_training_examples(
    examples: Iterable[TrainingExample],
) -> list[TrainingExample]:
    """Deduplicate prompt/target pairs while keeping the first weight."""

    result = []
    seen = set()
    for example in examples:
        key = (example.prompt, example.target)
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result
