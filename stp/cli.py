"""STP command-line interface."""

import argparse
import json
from pathlib import Path

from stp.core.config import load_config
from stp.data.declarations import build_declaration_artifact
from stp.self_play.stp_loop import evaluate, run, run_round


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for STP runs and evaluation."""

    parser = argparse.ArgumentParser(description="Single-GPU PyTorch STP")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Run or resume all rounds")
    run_parser.add_argument("--config", type=Path, required=True)

    round_parser = commands.add_parser("round", help="Run one HPC round")
    round_parser.add_argument("--config", type=Path, required=True)
    round_parser.add_argument("--round", type=int, required=True)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Evaluate a causal checkpoint",
    )
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)

    declarations_parser = commands.add_parser(
        "declarations",
        help="Build theorem declarations for the configured Lean project",
    )
    declarations_parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    """Run the selected STP command and print its result as JSON."""

    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "run":
        result = [state.__dict__ for state in run(config)]
    elif args.command == "round":
        result = run_round(config, args.round).__dict__
    elif args.command == "evaluate":
        result = evaluate(config, args.checkpoint)
    else:
        result = build_declaration_artifact(config)
    print(json.dumps(result, indent=2))
