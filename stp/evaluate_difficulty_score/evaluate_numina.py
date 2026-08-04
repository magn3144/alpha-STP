"""Evaluate one LLM and AlphaProof on the Numina theorem test set."""

import argparse
import time
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, cast

from stp.config import Config, ProverHandlerName, load_config
from stp.data import canonical_statement
from stp.generate import make_proof_requests
from stp.model import load_runtime
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
    load_records,
    read_jsonl,
    read_resumable_jsonl,
    write_jsonl,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"
EVALUATIONS_DIR = REPOSITORY / "data/evaluations"
LLM_ATTEMPTS = 32
ALPHAPROOF_ROLLOUTS = 250


@dataclass(frozen=True)
class EvaluationPaths:
    """Output paths for LLM results, AlphaProof results, and solver scores."""

    llm: Path
    alphaproof: Path
    scores: Path


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


def evaluation_paths(output_dir: Path) -> EvaluationPaths:
    """Build artifact paths from an evaluation directory and return them."""

    return EvaluationPaths(
        llm=output_dir / "llm_generations.jsonl",
        alphaproof=output_dir / "alphaproof_search_trees.jsonl",
        scores=output_dir / "difficulty_scores.jsonl",
    )


def statement_ids(problems: Sequence[Statement]) -> set[str]:
    """Collect statement IDs from problems and return them as a set."""

    return {problem.id for problem in problems}


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


def llm_solver_config(config: Config) -> Config:
    """Select fixed LLM evaluation settings and return the updated config."""

    return replace(
        config,
        solver=replace(
            config.solver,
            kind="llm",
            attempts_per_statement=LLM_ATTEMPTS,
        ),
    )


def alphaproof_solver_config(config: Config) -> Config:
    """Select fixed AlphaProof evaluation settings and return the updated config."""

    return replace(
        config,
        solver=replace(
            config.solver,
            kind="alphaproof",
            alphaproof_num_simulations=ALPHAPROOF_ROLLOUTS,
        ),
    )


def select_prover_handler(
    config: Config,
    handler: str | None,
) -> Config:
    """Apply an optional prover handler and return the selected config."""

    if handler is None:
        return config
    return replace(
        config,
        model=replace(
            config.model,
            prover_handler=cast(ProverHandlerName, handler),
        ),
    )


def run_llm_worker(
    requests: Sequence[ProofRequest],
    config: Config,
    model: str,
    tokenizer: str,
    output_path: Path,
) -> None:
    """Solve one problem with the LLM and write attempts from the child process."""

    runtime = load_runtime(model, tokenizer)
    attempts = solve_with_llm(requests, runtime, config)
    write_jsonl(output_path, attempts)


def solve_with_llm_process(
    requests: Sequence[ProofRequest],
    config: Config,
    model: str,
    tokenizer: str,
) -> list[SolveAttempt]:
    """Run one LLM problem in a spawned process and return its saved attempts."""

    with TemporaryDirectory(prefix="alpha-stp-llm-") as temporary:
        output_path = Path(temporary) / "attempts.jsonl"
        process = get_context("spawn").Process(
            target=run_llm_worker,
            args=(requests, config, model, tokenizer, output_path),
        )
        process.start()
        process.join()
        exitcode = process.exitcode
        process.close()
        if exitcode != 0:
            raise RuntimeError(
                f"LLM worker exited with status {exitcode}."
            )
        return load_records(output_path, SolveAttempt)


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


def solve_with_alphaproof_process(
    requests: Sequence[ProofRequest],
    config: Config,
) -> tuple[list[SolveAttempt], dict[str, Any]]:
    """Run one AlphaProof subprocess and return normalized attempts and raw result."""

    with TemporaryDirectory(prefix="alpha-stp-evaluation-") as temporary:
        artifact_dir = Path(temporary)
        attempts = solve_with_alphaproof(
            requests,
            config,
            artifact_dir,
        )
        raw_result = read_jsonl(artifact_dir / "alphaproof_results.jsonl")[0]
    return attempts, raw_result


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


def evaluate() -> None:
    """Run the complete resumable Numina evaluation from CLI inputs."""

    args = parse_args()
    config = load_config(args.config)
    config = select_prover_handler(config, args.llm_prover_handler)
    output_dir = evaluation_directory(args.name)
    model, tokenizer = model_paths(args, config)

    problems = load_problems(DATASET_PATH)
    problem_ids = statement_ids(problems)
    paths = evaluation_paths(output_dir)

    llm_results = llm_attempts_by_problem(
        load_results_by_problem(paths.llm, problem_ids)
    )
    alphaproof_results = alphaproof_attempts_by_problem(
        load_results_by_problem(paths.alphaproof, problem_ids)
    )
    saved_scores = score_keys(paths.scores, problem_ids)

    recover_missing_scores(
        paths.scores,
        llm_results,
        alphaproof_results,
        saved_scores,
    )

    llm_config = llm_solver_config(config)
    alphaproof_config = alphaproof_solver_config(config)
    llm_requests = make_proof_requests(problems, LLM_ATTEMPTS, config.run.seed)
    alphaproof_requests = make_proof_requests(problems, 1, config.run.seed)
    llm_requests_by_id = requests_by_problem(llm_requests)
    alphaproof_requests_by_id = requests_by_problem(alphaproof_requests)

    for problem in problems:
        if problem.id not in llm_results:
            llm_attempts = solve_with_llm_process(
                llm_requests_by_id[problem.id],
                llm_config,
                model,
                tokenizer,
            )
            save_llm_generations(
                paths.llm,
                problem,
                llm_attempts,
                llm_results,
            )
            append_score(paths.scores, llm_attempts, saved_scores)

        if problem.id not in alphaproof_results:
            alphaproof_attempts, raw_result = solve_with_alphaproof_process(
                alphaproof_requests_by_id[problem.id],
                alphaproof_config,
            )
            save_alphaproof_search(
                paths.alphaproof,
                problem,
                alphaproof_attempts,
                raw_result,
                alphaproof_results,
            )
            append_score(paths.scores, alphaproof_attempts, saved_scores)

    print(output_dir)


if __name__ == "__main__":
    evaluate()
