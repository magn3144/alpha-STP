"""Curriculum selection and conjecture scoring."""

import random
from collections import defaultdict
from typing import Sequence, TypeVar

import numpy as np

from stp.records import ConjectureAssessment, SolveAttempt, Statement

Candidate = TypeVar("Candidate")


def calculate_scores(
    attempts: Sequence[SolveAttempt],
) -> list[ConjectureAssessment]:
    """Calculate LLM solve rates or AlphaProof frontier scores."""

    grouped: dict[str, list[SolveAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.statement_id].append(attempt)

    assessments = []
    for statement_id, group in grouped.items():
        if group[0].solver == "alphaproof":
            if len(group) != 1:
                raise ValueError(
                    "AlphaProof must run exactly once per statement."
            )
            metric = {
                key: group[0].metrics[key]
                for key in (
                    "solved_frontier_nodes",
                    "total_frontier_nodes",
                    "solve_rate",
                )
            }
            assessments.append(
                ConjectureAssessment(
                    statement_id=statement_id,
                    method="alphaproof_hardest_subproblem",
                    score=float(metric["solve_rate"]),
                    attempts=1,
                    successes=int(group[0].status == "proved"),
                    metrics=metric,
                )
            )
            continue

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
    previous_scores: Sequence[ConjectureAssessment],
    limit: int,
    seed: int,
) -> list[Statement]:
    """Select difficult data plus a small replay of solved statements."""

    rates = {item.statement_id: item.score for item in previous_scores}
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


def unproved_statements(
    statements: Sequence[Statement],
    scores: Sequence[ConjectureAssessment],
) -> list[Statement]:
    """Return dataset statements with no successful proof."""

    proved = {
        score.statement_id for score in scores if score.successes > 0
    }
    return [statement for statement in statements if statement.id not in proved]


def wasserstein_matching(
    candidates: Sequence[Candidate],
    candidate_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_weights: Sequence[float],
    capacity: float = 3.0,
) -> list[tuple[Candidate, float]]:
    """Greedily match conjectures to the source distribution."""

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
