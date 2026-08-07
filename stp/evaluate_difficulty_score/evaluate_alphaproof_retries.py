"""Compare repeated AlphaProof searches with one large-search difficulty score."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from stp.core.config import Config, load_config
from stp.core.records import SolveAttempt, Statement, record_to_dict
from stp.data.storage import append_jsonl
from stp.evaluate_difficulty_score.data import (
    alphaproof_attempts_by_problem,
    load_problems,
    load_results_by_problem,
    requests_by_problem,
    solve_attempt,
    statement_ids,
)
from stp.evaluate_difficulty_score.evaluation_config import (
    DATASET_PATH,
    alphaproof_solver_config,
    evaluation_directory,
)
from stp.evaluate_difficulty_score.solver_processes import (
    SolverProcessError,
    solve_with_alphaproof_process,
)
from stp.self_play.generate import make_proof_requests


RETRY_COUNT = 16
RETRY_ROLLOUTS = 200
LARGE_SEARCH_ROLLOUTS = 1024


def parse_args() -> argparse.Namespace:
    """Parse experiment inputs and return the selected CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare 16 AlphaProof retries with one large-search difficulty "
            "score on Numina Lean theorems."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--name",
        help="Output folder name under data/evaluations; defaults to a timestamp.",
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        help="Zero-based Numina problem index; defaults to the complete dataset.",
    )
    args = parser.parse_args()
    if args.problem_index is not None and args.problem_index < 0:
        parser.error("--problem-index must be non-negative")
    return args


def solver_config(config: Config, rollouts: int) -> Config:
    """Set the AlphaProof rollout budget and return a new solver config."""

    alphaproof_config = alphaproof_solver_config(config)
    return replace(
        alphaproof_config,
        solver=replace(
            alphaproof_config.solver,
            alphaproof_num_simulations=rollouts,
        ),
    )


def retry_attempts_by_problem(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[SolveAttempt]]:
    """Deserialize retry records and return attempts grouped by problem ID."""

    return {
        problem_id: [solve_attempt(value) for value in record["attempts"]]
        for problem_id, record in records.items()
    }


def save_retry_attempts(
    path: Path,
    problem: Statement,
    attempts: list[SolveAttempt],
    results: dict[str, list[SolveAttempt]],
) -> None:
    """Append one problem's retry attempts and update in-memory results."""

    append_jsonl(
        path,
        {
            "problem_id": problem.id,
            "searches": RETRY_COUNT,
            "rollouts_per_search": RETRY_ROLLOUTS,
            "attempts": [record_to_dict(attempt) for attempt in attempts],
        },
    )
    results[problem.id] = attempts


def save_large_search(
    path: Path,
    problem: Statement,
    attempts: list[SolveAttempt],
    raw_result: dict[str, Any],
    results: dict[str, list[SolveAttempt]],
) -> None:
    """Append one large search tree and update in-memory results."""

    append_jsonl(
        path,
        {
            "problem_id": problem.id,
            "request_id": raw_result["request_id"],
            "rollouts": LARGE_SEARCH_ROLLOUTS,
            "attempt": record_to_dict(attempts[0]),
            "tree": raw_result["tree"],
        },
    )
    results[problem.id] = attempts


def summed_times(attempts: Sequence[SolveAttempt]) -> tuple[float, float]:
    """Sum solver and verification durations and return both totals."""

    solver_seconds = sum(attempt.duration_seconds for attempt in attempts)
    verification_seconds = sum(
        attempt.verify_seconds for attempt in attempts
    )
    return solver_seconds, verification_seconds


def comparison_record(
    retry_attempts: Sequence[SolveAttempt],
    large_attempts: Sequence[SolveAttempt],
) -> dict[str, Any]:
    """Compare retry success with the large metric and return one record."""

    retry_total = sum(attempt.multiplicity for attempt in retry_attempts)
    if retry_total != RETRY_COUNT:
        raise ValueError(
            f"Expected {RETRY_COUNT} retry searches, found {retry_total}."
        )
    if len(large_attempts) != 1:
        raise ValueError("Expected exactly one large AlphaProof search.")

    retry_successes = sum(
        attempt.multiplicity
        for attempt in retry_attempts
        if attempt.status == "proved"
    )
    retry_solver_seconds, retry_verification_seconds = summed_times(
        retry_attempts
    )
    large_solver_seconds, large_verification_seconds = summed_times(
        large_attempts
    )
    large_attempt = large_attempts[0]
    metrics = large_attempt.metrics
    return {
        "problem_id": large_attempt.statement_id,
        "retry_method": "solve_rate",
        "retry_searches": retry_total,
        "retry_rollouts_per_search": RETRY_ROLLOUTS,
        "retry_successes": retry_successes,
        "retry_solve_rate": retry_successes / retry_total,
        "retry_solver_seconds": retry_solver_seconds,
        "retry_verification_seconds": retry_verification_seconds,
        "retry_total_seconds": (
            retry_solver_seconds + retry_verification_seconds
        ),
        "large_search_method": "alphaproof_hardest_subproblem",
        "large_search_rollouts": LARGE_SEARCH_ROLLOUTS,
        "large_search_status": large_attempt.status,
        "large_search_solved_frontier_nodes": metrics[
            "solved_frontier_nodes"
        ],
        "large_search_total_frontier_nodes": metrics[
            "total_frontier_nodes"
        ],
        "large_search_solve_rate": metrics["solve_rate"],
        "large_search_solver_seconds": large_solver_seconds,
        "large_search_verification_seconds": large_verification_seconds,
        "large_search_total_seconds": (
            large_solver_seconds + large_verification_seconds
        ),
    }


def append_comparison(
    path: Path,
    retry_attempts: Sequence[SolveAttempt],
    large_attempts: Sequence[SolveAttempt],
    saved_problem_ids: set[str],
) -> None:
    """Append a comparison unless its problem ID is already saved."""

    problem_id = large_attempts[0].statement_id
    if problem_id in saved_problem_ids:
        return
    append_jsonl(path, comparison_record(retry_attempts, large_attempts))
    saved_problem_ids.add(problem_id)


def recover_comparisons(
    path: Path,
    retry_results: dict[str, list[SolveAttempt]],
    large_results: dict[str, list[SolveAttempt]],
    saved_problem_ids: set[str],
) -> None:
    """Create comparisons missing for already completed paired searches."""

    for problem_id in retry_results:
        if problem_id not in large_results:
            continue
        append_comparison(
            path,
            retry_results[problem_id],
            large_results[problem_id],
            saved_problem_ids,
        )


def select_problems(problem_index: int | None) -> list[Statement]:
    """Load Numina problems and return the requested evaluation subset."""

    problems = load_problems(DATASET_PATH)
    if problem_index is None:
        return problems
    if problem_index >= len(problems):
        raise IndexError(
            f"Problem index {problem_index} is outside the dataset "
            f"of {len(problems)} problems."
        )
    return [problems[problem_index]]


def evaluate() -> None:
    """Run the complete resumable AlphaProof retry comparison."""

    args = parse_args()
    config = load_config(args.config)
    output_dir = evaluation_directory(args.name)
    retry_path = output_dir / "alphaproof_retries.jsonl"
    large_path = output_dir / "alphaproof_large_search_trees.jsonl"
    comparison_path = output_dir / "alphaproof_retry_comparison.jsonl"

    problems = select_problems(args.problem_index)
    problem_ids = statement_ids(problems)
    retry_results = retry_attempts_by_problem(
        load_results_by_problem(retry_path, problem_ids)
    )
    large_results = alphaproof_attempts_by_problem(
        load_results_by_problem(large_path, problem_ids)
    )
    comparison_results = load_results_by_problem(
        comparison_path,
        problem_ids,
    )
    saved_comparisons = set(comparison_results)
    recover_comparisons(
        comparison_path,
        retry_results,
        large_results,
        saved_comparisons,
    )

    retry_config = solver_config(config, RETRY_ROLLOUTS)
    large_config = solver_config(config, LARGE_SEARCH_ROLLOUTS)
    requests = make_proof_requests(
        problems,
        RETRY_COUNT + 1,
        config.run.seed,
    )
    requests_by_id = requests_by_problem(requests)

    for problem in problems:
        problem_requests = requests_by_id[problem.id]
        if problem.id not in retry_results:
            try:
                attempts, _ = solve_with_alphaproof_process(
                    problem_requests[:RETRY_COUNT],
                    retry_config,
                )
            except SolverProcessError as error:
                print(
                    f"AlphaProof retries failed for {problem.id}: {error}",
                    flush=True,
                )
            else:
                save_retry_attempts(
                    retry_path,
                    problem,
                    attempts,
                    retry_results,
                )

        if problem.id not in large_results:
            try:
                attempts, raw_result = solve_with_alphaproof_process(
                    problem_requests[RETRY_COUNT:],
                    large_config,
                )
            except SolverProcessError as error:
                print(
                    f"Large AlphaProof search failed for {problem.id}: {error}",
                    flush=True,
                )
            else:
                save_large_search(
                    large_path,
                    problem,
                    attempts,
                    raw_result,
                    large_results,
                )

        if problem.id in retry_results and problem.id in large_results:
            append_comparison(
                comparison_path,
                retry_results[problem.id],
                large_results[problem.id],
                saved_comparisons,
            )

    print(output_dir)


if __name__ == "__main__":
    evaluate()
