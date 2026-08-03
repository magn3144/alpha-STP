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
from stp.records import ConjectureAssessment, SolveAttempt, Statement
from stp.scoring import calculate_scores
from stp.solvers import solve_with_alphaproof, solve_with_llm
from stp.storage import read_jsonl, write_jsonl


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
    """Resolve one output name and return a new folder under data/evaluations."""

    folder_name = name or time.strftime("%Y%m%d-%H%M%S")
    if Path(folder_name).name != folder_name:
        raise ValueError("Evaluation name must be a single folder name.")
    directory = EVALUATIONS_DIR / folder_name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def model_paths(args: argparse.Namespace, config: Config) -> tuple[str, str]:
    """Resolve CLI and config choices and return model and tokenizer paths."""

    model = args.llm_model or config.model.name
    tokenizer = args.llm_tokenizer or (
        model if args.llm_model is not None else config.model.tokenizer
    )
    return model, tokenizer


def assessments_by_id(
    assessments: Sequence[ConjectureAssessment],
) -> dict[str, ConjectureAssessment]:
    """Index per-problem assessments and return them by statement ID."""

    return {assessment.statement_id: assessment for assessment in assessments}


def timing_by_id(
    attempts: Sequence[SolveAttempt],
) -> dict[str, tuple[float, float]]:
    """Sum solver and verification seconds and return them by statement ID."""

    timings: dict[str, tuple[float, float]] = {}
    for attempt in attempts:
        solver_seconds, verification_seconds = timings.get(
            attempt.statement_id,
            (0.0, 0.0),
        )
        timings[attempt.statement_id] = (
            solver_seconds + attempt.duration_seconds,
            verification_seconds + attempt.verify_seconds,
        )
    return timings


def combine_scores(
    problems: Sequence[Statement],
    llm_scores: Sequence[ConjectureAssessment],
    alphaproof_scores: Sequence[ConjectureAssessment],
    llm_attempts: Sequence[SolveAttempt],
    alphaproof_attempts: Sequence[SolveAttempt],
) -> list[dict[str, Any]]:
    """Combine both solver assessments and return one score record per problem."""

    llm_by_id = assessments_by_id(llm_scores)
    alphaproof_by_id = assessments_by_id(alphaproof_scores)
    llm_timing_by_id = timing_by_id(llm_attempts)
    alphaproof_timing_by_id = timing_by_id(alphaproof_attempts)
    scores = []
    for problem in problems:
        llm = llm_by_id[problem.id]
        alphaproof = alphaproof_by_id[problem.id]
        llm_solver_seconds, llm_verification_seconds = llm_timing_by_id[
            problem.id
        ]
        alphaproof_solver_seconds, alphaproof_verification_seconds = (
            alphaproof_timing_by_id[problem.id]
        )
        if llm.attempts != LLM_ATTEMPTS:
            raise ValueError(
                f"Expected {LLM_ATTEMPTS} LLM attempts for {problem.id}, "
                f"got {llm.attempts}."
            )
        scores.append(
            {
                "problem_id": problem.id,
                "llm_score": llm.score,
                "llm_attempts": llm.attempts,
                "llm_successes": llm.successes,
                "llm_solver_seconds": llm_solver_seconds,
                "llm_verification_seconds": llm_verification_seconds,
                "llm_total_seconds": (
                    llm_solver_seconds + llm_verification_seconds
                ),
                "alphaproof_score": alphaproof.score,
                "alphaproof_solved": alphaproof.successes > 0,
                "alphaproof_solver_seconds": alphaproof_solver_seconds,
                "alphaproof_verification_seconds": (
                    alphaproof_verification_seconds
                ),
                "alphaproof_total_seconds": (
                    alphaproof_solver_seconds
                    + alphaproof_verification_seconds
                ),
                **alphaproof.metrics,
            }
        )
    return scores


def evaluate(config: Config, model: str, tokenizer: str, output_dir: Path) -> None:
    """Run both solvers and write proof, tree, and score JSONL artifacts."""

    problems = load_problems(DATASET_PATH)

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
    runtime = load_runtime(model, tokenizer)
    try:
        llm_attempts = solve_with_llm(llm_requests, runtime, llm_config)
    finally:
        unload_runtime(runtime)
    write_jsonl(output_dir / "llm_proof_attempts.jsonl", llm_attempts)
    llm_scores = calculate_scores(llm_attempts)

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
    with TemporaryDirectory(prefix="alpha-stp-evaluation-") as temporary:
        artifact_dir = Path(temporary)
        alphaproof_attempts = solve_with_alphaproof(
            alphaproof_requests,
            alphaproof_config,
            artifact_dir,
        )
        raw_results = read_jsonl(artifact_dir / "alphaproof_results.jsonl")

    problem_id_by_request = {
        request.id: request.statement_id for request in alphaproof_requests
    }
    trees = [
        {
            "problem_id": problem_id_by_request[str(result["request_id"])],
            "request_id": result["request_id"],
            "tree": result["tree"],
        }
        for result in raw_results
    ]
    write_jsonl(output_dir / "alphaproof_search_trees.jsonl", trees)

    alphaproof_scores = calculate_scores(alphaproof_attempts)
    scores = combine_scores(
        problems,
        llm_scores,
        alphaproof_scores,
        llm_attempts,
        alphaproof_attempts,
    )
    write_jsonl(output_dir / "difficulty_scores.jsonl", scores)


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
