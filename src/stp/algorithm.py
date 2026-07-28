import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence, TypeVar

import numpy as np

from stp.records import (
    Conjecture,
    ConjectureAssessment,
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


def select_conjecture_sources(
    statements: Sequence[Statement | Conjecture],
    attempts: Sequence[SolveAttempt],
    assessments: Sequence[ConjectureAssessment],
    limit: int,
    seed: int,
) -> list[tuple[Statement | Conjecture, str]]:
    """Select easy proven statements and one verified proof for conjecturing."""

    by_id = {statement.id: statement for statement in statements}
    rates = {assessment.statement_id: assessment.score for assessment in assessments}
    sources = []
    used = set()
    for attempt in reversed(attempts):
        if (
            attempt.status == "proved"
            and attempt.proof is not None
            and rates.get(attempt.statement_id, 0.0) > 0.35
            and attempt.statement_id not in used
            and attempt.statement_id in by_id
        ):
            used.add(attempt.statement_id)
            sources.append((by_id[attempt.statement_id], attempt.proof))
    rng = random.Random(seed)
    rng.shuffle(sources)
    return sources[:limit]


def wasserstein_matching(
    candidates: Sequence[Candidate],
    candidate_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_weights: Sequence[float],
    capacity: float = 4.0,
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
        count = min(int(target_weight), len(candidates))
        scores = np.where(active, similarities[:, target_index], -np.inf)
        chosen = np.argpartition(scores, -count)[-count:]
        for candidate_index in chosen:
            weights[candidate_index] += len(candidates) / total_weight
            if weights[candidate_index] >= capacity:
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
    conjecture_threshold: float,
) -> list[TrainingExample]:
    """Build weighted proof and easy-to-hard examples from verified results."""

    by_id = {statement.id: statement for statement in statements}
    rates = {assessment.statement_id: assessment.score for assessment in assessments}
    success_counts: dict[str, int] = defaultdict(int)
    for attempt in attempts:
        if attempt.status == "proved":
            success_counts[attempt.statement_id] += attempt.multiplicity

    examples = []
    seen = set()
    conjecture_seen = set()
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
        if (
            isinstance(statement, Conjecture)
            and 0.0 < rates[statement.id] <= conjecture_threshold
            and statement.id not in conjecture_seen
        ):
            conjecture_seen.add(statement.id)
            target = (
                f"\n{statement.statement.removesuffix(':= by').strip()}\n"
                f"{END_THM}"
            )
            examples.append(
                TrainingExample(
                    prompt=conjecturer_training_prompt(
                        statement.shared_lemma_statement,
                        statement.easy_statement,
                        statement.easy_proof,
                    ),
                    target=target,
                    weight=1.0,
                    kind="conjecture",
                    statement_id=statement.id,
                    round=round_index,
                )
            )
    return examples


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
