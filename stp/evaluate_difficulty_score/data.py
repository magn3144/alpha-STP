"""Problem and artifact data operations for Numina evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from stp.data import canonical_statement
from stp.records import (
    ProofRequest,
    SolveAttempt,
    SolveStatus,
    Statement,
    record_to_dict,
)
from stp.storage import append_jsonl, read_jsonl, read_resumable_jsonl


@dataclass(frozen=True)
class EvaluationPaths:
    """Output paths for LLM results, AlphaProof results, and solver scores."""

    llm: Path
    alphaproof: Path
    scores: Path


def load_problems(path: Path) -> list[Statement]:
    """Load Numina JSONL from path and return canonical evaluation statements."""

    problems = []
    for value in read_jsonl(path):
        problems.append(
            Statement(
                id=str(value["id"]),
                statement=canonical_statement(str(value["theorem"])),
                header=None,
                labels=("numina_sft_evaluation", "test"),
                matching_weight=1.0,
                source=str(path),
            )
        )
    return problems


def statement_ids(problems: Sequence[Statement]) -> set[str]:
    """Collect statement IDs from problems and return them as a set."""

    return {problem.id for problem in problems}


def evaluation_paths(output_dir: Path) -> EvaluationPaths:
    """Build artifact paths from an evaluation directory and return them."""

    return EvaluationPaths(
        llm=output_dir / "llm_generations.jsonl",
        alphaproof=output_dir / "alphaproof_search_trees.jsonl",
        scores=output_dir / "difficulty_scores.jsonl",
    )


def requests_by_problem(
    requests: Sequence[ProofRequest],
) -> dict[str, list[ProofRequest]]:
    """Group proof requests by statement ID and return them in input order."""

    grouped: dict[str, list[ProofRequest]] = {}
    for request in requests:
        grouped.setdefault(request.statement_id, []).append(request)
    return grouped


def solve_attempt(value: dict[str, Any]) -> SolveAttempt:
    """Convert one serialized solve-attempt mapping into a record."""

    return SolveAttempt(
        request_id=str(value["request_id"]),
        statement_id=str(value["statement_id"]),
        attempt=int(value["attempt"]),
        solver=str(value["solver"]),
        seed=int(value["seed"]),
        status=cast(SolveStatus, value["status"]),
        proof=cast(str | None, value["proof"]),
        duration_seconds=float(value["duration_seconds"]),
        generated_tokens=int(value["generated_tokens"]),
        verify_seconds=float(value["verify_seconds"]),
        invoked_lemmas=tuple(value["invoked_lemmas"]),
        multiplicity=int(value["multiplicity"]),
        metrics=dict(value["metrics"]),
        raw_output=cast(str | None, value["raw_output"]),
    )


def load_results_by_problem(
    path: Path,
    problem_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load completed JSONL results and index them by problem ID."""

    records: dict[str, dict[str, Any]] = {}
    for value in read_resumable_jsonl(path):
        problem_id = str(value["problem_id"])
        if problem_id not in problem_ids:
            raise ValueError(f"Unknown problem ID in {path}: {problem_id}")
        if problem_id in records:
            raise ValueError(f"Duplicate problem ID in {path}: {problem_id}")
        records[problem_id] = value
    return records


def llm_attempts_by_problem(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[SolveAttempt]]:
    """Deserialize saved LLM records and return attempts by problem ID."""

    return {
        problem_id: [solve_attempt(value) for value in record["attempts"]]
        for problem_id, record in records.items()
    }


def alphaproof_attempts_by_problem(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[SolveAttempt]]:
    """Deserialize saved AlphaProof records and return attempts by problem ID."""

    return {
        problem_id: [solve_attempt(record["attempt"])]
        for problem_id, record in records.items()
    }


def save_llm_generations(
    path: Path,
    problem: Statement,
    attempts: list[SolveAttempt],
    results: dict[str, list[SolveAttempt]],
) -> None:
    """Append one problem's LLM generations and update completed results."""

    append_jsonl(
        path,
        {
            "problem_id": problem.id,
            "attempts": [record_to_dict(attempt) for attempt in attempts],
        },
    )
    results[problem.id] = attempts


def save_alphaproof_search(
    path: Path,
    problem: Statement,
    attempts: list[SolveAttempt],
    raw_result: dict[str, Any],
    results: dict[str, list[SolveAttempt]],
) -> None:
    """Append one AlphaProof search tree and update completed results."""

    append_jsonl(
        path,
        {
            "problem_id": problem.id,
            "request_id": raw_result["request_id"],
            "attempt": record_to_dict(attempts[0]),
            "tree": raw_result["tree"],
        },
    )
    results[problem.id] = attempts
