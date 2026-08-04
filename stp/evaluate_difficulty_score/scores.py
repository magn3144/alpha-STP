"""Incremental score operations for Numina evaluation."""

from pathlib import Path
from typing import Any, Sequence

from stp.evaluate_difficulty_score.evaluation_config import LLM_ATTEMPTS
from stp.records import SolveAttempt
from stp.scoring import calculate_scores
from stp.storage import append_jsonl, read_resumable_jsonl


def score_record(attempts: Sequence[SolveAttempt]) -> dict[str, Any]:
    """Calculate one problem-solver score and return its serialized fields."""

    assessment = calculate_scores(attempts)[0]
    if attempts[0].solver == "llm" and assessment.attempts != LLM_ATTEMPTS:
        raise ValueError(
            f"Expected {LLM_ATTEMPTS} LLM attempts for {assessment.statement_id}, "
            f"got {assessment.attempts}."
        )
    solver_seconds = sum(attempt.duration_seconds for attempt in attempts)
    verification_seconds = sum(attempt.verify_seconds for attempt in attempts)
    return {
        "problem_id": assessment.statement_id,
        "solver": attempts[0].solver,
        "method": assessment.method,
        "score": assessment.score,
        "attempts": assessment.attempts,
        "successes": assessment.successes,
        "solver_seconds": solver_seconds,
        "verification_seconds": verification_seconds,
        "total_seconds": solver_seconds + verification_seconds,
        **assessment.metrics,
    }


def score_keys(path: Path, problem_ids: set[str]) -> set[tuple[str, str]]:
    """Load saved score identities and return unique problem-solver keys."""

    keys: set[tuple[str, str]] = set()
    for value in read_resumable_jsonl(path):
        problem_id = str(value["problem_id"])
        solver = str(value["solver"])
        if problem_id not in problem_ids:
            raise ValueError(f"Unknown problem ID in {path}: {problem_id}")
        if solver not in ("llm", "alphaproof"):
            raise ValueError(f"Unknown solver in {path}: {solver}")
        key = (problem_id, solver)
        if key in keys:
            raise ValueError(f"Duplicate score in {path}: {problem_id}, {solver}")
        keys.add(key)
    return keys


def append_score(
    path: Path,
    attempts: Sequence[SolveAttempt],
    saved_keys: set[tuple[str, str]],
) -> None:
    """Append one problem-solver score unless it is already saved."""

    key = (attempts[0].statement_id, attempts[0].solver)
    if key in saved_keys:
        return
    append_jsonl(path, score_record(attempts))
    saved_keys.add(key)


def recover_missing_scores(
    path: Path,
    llm_results: dict[str, list[SolveAttempt]],
    alphaproof_results: dict[str, list[SolveAttempt]],
    saved_scores: set[tuple[str, str]],
) -> None:
    """Append scores missing for solver results saved before an interruption."""

    for attempts_by_id in (llm_results, alphaproof_results):
        for attempts in attempts_by_id.values():
            append_score(path, attempts, saved_scores)
