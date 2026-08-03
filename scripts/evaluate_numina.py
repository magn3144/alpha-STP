"""Evaluate one LLM and AlphaProof on the Numina theorem test set."""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, cast

from stp.config import Config, ProverHandlerName, load_config
from stp.data import canonical_statement
from stp.generate import make_proof_requests
from stp.model import load_runtime, unload_runtime
from stp.records import (
    ProofRequest,
    SolveAttempt,
    SolveStatus,
    Statement,
    record_to_dict,
)
from stp.scoring import calculate_scores
from stp.solvers import solve_with_alphaproof, solve_with_llm
from stp.storage import (
    append_jsonl,
    read_jsonl,
    read_resumable_jsonl,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"
EVALUATIONS_DIR = REPOSITORY / "data/evaluations"
LLM_ATTEMPTS = 32
ALPHAPROOF_ROLLOUTS = 250


def parse_args() -> argparse.Namespace:
    """Parse the config and LLM inputs and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate an LLM and AlphaProof on Numina Lean theorems.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--llm-model",
        help="Hugging Face model name or local checkpoint; defaults to config.model.name.",
    )
    parser.add_argument(
        "--llm-tokenizer",
        help="Tokenizer name or path; defaults to the selected LLM model.",
    )
    parser.add_argument(
        "--llm-prover-handler",
        choices=("stp", "kimina_numina"),
        help="Prompt and answer parser; defaults to config.model.prover_handler.",
    )
    parser.add_argument(
        "--name",
        help="Output folder name under data/evaluations; defaults to a timestamp.",
    )
    return parser.parse_args()


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


def evaluation_directory(name: str | None) -> Path:
    """Resolve one output name and return its folder under data/evaluations."""

    folder_name = name or time.strftime("%Y%m%d-%H%M%S")
    if Path(folder_name).name != folder_name:
        raise ValueError("Evaluation name must be a single folder name.")
    directory = EVALUATIONS_DIR / folder_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def model_paths(args: argparse.Namespace, config: Config) -> tuple[str, str]:
    """Resolve CLI and config choices and return model and tokenizer paths."""

    model = args.llm_model or config.model.name
    tokenizer = args.llm_tokenizer or (
        model if args.llm_model is not None else config.model.tokenizer
    )
    return model, tokenizer


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


def records_by_problem(
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


def score_record(
    attempts: Sequence[SolveAttempt],
) -> dict[str, Any]:
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


def score_keys(
    path: Path,
    problem_ids: set[str],
) -> set[tuple[str, str]]:
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


def evaluate(config: Config, model: str, tokenizer: str, output_dir: Path) -> None:
    """Run resumable per-problem solver stages and write JSONL artifacts."""

    problems = load_problems(DATASET_PATH)
    problem_ids = {problem.id for problem in problems}
    llm_path = output_dir / "llm_generations.jsonl"
    alphaproof_path = output_dir / "alphaproof_search_trees.jsonl"
    scores_path = output_dir / "difficulty_scores.jsonl"
    llm_attempts_by_id = llm_attempts_by_problem(
        records_by_problem(llm_path, problem_ids)
    )
    alphaproof_attempts_by_id = alphaproof_attempts_by_problem(
        records_by_problem(alphaproof_path, problem_ids)
    )
    saved_score_keys = score_keys(scores_path, problem_ids)
    for attempts_by_id in (llm_attempts_by_id, alphaproof_attempts_by_id):
        for attempts in attempts_by_id.values():
            append_score(scores_path, attempts, saved_score_keys)

    llm_config = replace(
        config,
        solver=replace(
            config.solver,
            kind="llm",
            attempts_per_statement=LLM_ATTEMPTS,
        ),
    )
    llm_requests = make_proof_requests(
        problems,
        LLM_ATTEMPTS,
        config.run.seed,
    )
    llm_requests_by_id = requests_by_problem(llm_requests)
    pending_llm = [
        problem for problem in problems if problem.id not in llm_attempts_by_id
    ]
    if pending_llm:
        runtime = load_runtime(model, tokenizer)
        try:
            for problem in pending_llm:
                attempts = solve_with_llm(
                    llm_requests_by_id[problem.id],
                    runtime,
                    llm_config,
                )
                record = {
                    "problem_id": problem.id,
                    "attempts": [record_to_dict(attempt) for attempt in attempts],
                }
                append_jsonl(llm_path, record)
                llm_attempts_by_id[problem.id] = attempts
                append_score(scores_path, attempts, saved_score_keys)
        finally:
            unload_runtime(runtime)

    alphaproof_config = replace(
        config,
        solver=replace(
            config.solver,
            kind="alphaproof",
            alphaproof_num_simulations=ALPHAPROOF_ROLLOUTS,
        ),
    )
    alphaproof_requests = make_proof_requests(
        problems,
        1,
        config.run.seed,
    )
    alphaproof_requests_by_id = requests_by_problem(alphaproof_requests)
    for problem in problems:
        if problem.id in alphaproof_attempts_by_id:
            continue
        requests = alphaproof_requests_by_id[problem.id]
        with TemporaryDirectory(prefix="alpha-stp-evaluation-") as temporary:
            artifact_dir = Path(temporary)
            attempts = solve_with_alphaproof(
                requests,
                alphaproof_config,
                artifact_dir,
            )
            raw_result = read_jsonl(
                artifact_dir / "alphaproof_results.jsonl"
            )[0]
        record = {
            "problem_id": problem.id,
            "request_id": raw_result["request_id"],
            "attempt": record_to_dict(attempts[0]),
            "tree": raw_result["tree"],
        }
        append_jsonl(alphaproof_path, record)
        alphaproof_attempts_by_id[problem.id] = attempts
        append_score(scores_path, attempts, saved_score_keys)


def main() -> None:
    """Load configuration and run the complete Numina evaluation."""

    args = parse_args()
    config = load_config(args.config)
    if args.llm_prover_handler is not None:
        config = replace(
            config,
            model=replace(
                config.model,
                prover_handler=cast(
                    ProverHandlerName,
                    args.llm_prover_handler,
                ),
            ),
        )
    model, tokenizer = model_paths(args, config)
    output_dir = evaluation_directory(args.name)
    evaluate(config, model, tokenizer, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
