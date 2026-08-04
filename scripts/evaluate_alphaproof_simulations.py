"""Compare AlphaProof search budgets with proof-tree difficulty scores."""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from stp.core.config import Config, load_config
from stp.data.datasets import canonical_statement
from stp.self_play.generate import make_proof_requests
from stp.core.records import Statement
from stp.self_play.scoring import calculate_scores
from stp.proving.solvers import solve_with_alphaproof
from stp.data.storage import read_jsonl, write_jsonl


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"
EVALUATIONS_DIR = REPOSITORY / "data/evaluations"
ALPHAPROOF_RUN_DIR = (
    REPOSITORY.parent / "delta-proof/data/runs/sft_codet5p_770m_v100_32gb"
)
SIMULATION_BUDGETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs and return the config, problem, and output choices."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure one problem's AlphaProof difficulty score across search "
            "budgets."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="Zero-based problem index in the Numina test split (default: 0).",
    )
    parser.add_argument(
        "--name",
        help="Output folder name under data/evaluations; defaults to a timestamp.",
    )
    args = parser.parse_args()
    if args.problem_index < 0:
        parser.error("--problem-index must be non-negative")
    return args


def load_problem(path: Path, index: int) -> Statement:
    """Load a JSONL dataset path and return the problem at a zero-based index."""

    values = read_jsonl(path)
    if index >= len(values):
        raise IndexError(
            f"Problem index {index} is outside the dataset of {len(values)} problems."
        )
    value = values[index]
    return Statement(
        id=str(value["id"]),
        statement=canonical_statement(str(value["theorem"])),
        header=None,
        labels=("numina_sft_evaluation", "test"),
        matching_weight=1.0,
        source=str(path),
    )


def evaluation_directory(name: str | None) -> Path:
    """Resolve an output name and return a new directory under evaluations."""

    folder_name = name or time.strftime("alphaproof-simulations-%Y%m%d-%H%M%S")
    if Path(folder_name).name != folder_name:
        raise ValueError("Evaluation name must be a single folder name.")
    directory = EVALUATIONS_DIR / folder_name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def run_budget(
    problem: Statement,
    config: Config,
    simulations: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one budget on a problem and return its tree-derived score record."""

    budget_config = replace(
        config,
        solver=replace(
            config.solver,
            kind="alphaproof",
            alphaproof_num_simulations=simulations,
        ),
    )
    requests = make_proof_requests([problem], 1, config.run.seed)
    artifact_dir = output_dir / f"simulations-{simulations}"
    artifact_dir.mkdir()
    attempts = solve_with_alphaproof(requests, budget_config, artifact_dir)
    attempt = attempts[0]
    assessment = calculate_scores(attempts)[0]
    return {
        "problem_id": problem.id,
        "simulations": simulations,
        "alphaproof_score": assessment.score,
        "metric_method": assessment.method,
        "status": attempt.status,
        **assessment.metrics,
        "solver_seconds": attempt.duration_seconds,
        "verification_seconds": attempt.verify_seconds,
        "total_seconds": attempt.duration_seconds + attempt.verify_seconds,
    }


def save_plot(records: list[dict[str, Any]], output_path: Path) -> None:
    """Plot simulation budgets and difficulty metrics and save a PNG file."""

    simulations = [int(record["simulations"]) for record in records]
    scores = [float(record["alphaproof_score"]) for record in records]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(simulations, scores, marker="o")
    axis.set_xscale("log", base=2)
    axis.set_xticks(simulations, labels=[str(value) for value in simulations])
    axis.set_xlabel("AlphaProof simulations")
    axis.set_ylabel("Difficulty metric (frontier solve rate)")
    axis.set_title("AlphaProof difficulty metric by search budget")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate(config: Config, problem: Statement, output_dir: Path) -> None:
    """Evaluate all fixed budgets and save JSONL metrics and a line plot."""

    records = []
    metrics_path = output_dir / "difficulty_by_simulations.jsonl"
    for simulations in SIMULATION_BUDGETS:
        record = run_budget(problem, config, simulations, output_dir)
        records.append(record)
        write_jsonl(metrics_path, records)
        print(
            f"{simulations} simulations: "
            f"alphaproof_score={record['alphaproof_score']:.6f}"
        )
    save_plot(records, output_dir / "difficulty_by_simulations.png")


def main() -> None:
    """Load the selected problem and run the simulation-budget evaluation."""

    args = parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        solver=replace(
            config.solver,
            alphaproof_run_dir=ALPHAPROOF_RUN_DIR,
        ),
    )
    problem = load_problem(DATASET_PATH, args.problem_index)
    output_dir = evaluation_directory(args.name)
    evaluate(config, problem, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
