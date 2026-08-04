"""Measure AlphaProof difficulty for one Numina evaluation problem."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from stp.core.config import load_config
from stp.data.datasets import alphaproof_theorem, canonical_statement
from stp.self_play.generate import make_proof_requests
from stp.core.records import Statement
from stp.proving.search_metrics import hardest_subproblem_solve_rate
from stp.data.storage import read_jsonl, write_jsonl


REPOSITORY = Path(__file__).resolve().parents[1]
ALPHAPROOF_REPOSITORY = REPOSITORY.parent / "delta-proof"
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs and return the selected config, problem, and budget."""

    parser = argparse.ArgumentParser(
        description="Run AlphaProof difficulty scoring on one Numina problem.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="Zero-based problem index in the Numina test split (default: 0).",
    )
    args = parser.parse_args()
    if args.simulations < 1:
        parser.error("--simulations must be positive")
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


def main() -> None:
    """Run one AlphaProof search and print its difficulty and timing metrics."""

    args = parse_args()
    os.environ["PYTHONPATH"] = str(ALPHAPROOF_REPOSITORY)
    config = load_config(args.config)
    config = replace(
        config,
        solver=replace(
            config.solver,
            kind="alphaproof",
            alphaproof_command=(
                sys.executable,
                *config.solver.alphaproof_command[1:],
            ),
            alphaproof_num_simulations=args.simulations,
        ),
    )
    problem = load_problem(DATASET_PATH, args.problem_index)
    request = make_proof_requests([problem], 1, config.run.seed)[0]

    with TemporaryDirectory(prefix="alpha-stp-difficulty-") as temporary:
        artifact_dir = Path(temporary)
        input_path = artifact_dir / "request.jsonl"
        output_path = artifact_dir / "result.jsonl"
        write_jsonl(
            input_path,
            [
                {
                    "request_id": request.id,
                    "theorem": alphaproof_theorem(request.statement),
                    "header": request.header,
                    "seed": request.seed,
                }
            ],
        )
        command = [
            *config.solver.alphaproof_command,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--run-dir",
            str(config.solver.alphaproof_run_dir),
            "--lean-project",
            str(config.lean.project_dir),
            "--num-simulations",
            str(args.simulations),
            "--num-sampled-actions",
            str(config.solver.alphaproof_num_sampled_actions),
            "--no-stop-on-solution",
        ]
        for module in config.lean.imports:
            command.extend(("--import", module))
        subprocess.run(
            command,
            check=True,
            timeout=config.solver.alphaproof_timeout_seconds,
        )
        result = read_jsonl(output_path)[0]

    metrics = hardest_subproblem_solve_rate(result["tree"])
    print(
        json.dumps(
            {
                "problem_index": args.problem_index,
                "problem_id": problem.id,
                "simulations": args.simulations,
                "status": result["status"],
                "proof": result.get("proof"),
                "alphaproof_score": metrics["solve_rate"],
                **metrics,
                "solver_seconds": float(result["duration_seconds"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
