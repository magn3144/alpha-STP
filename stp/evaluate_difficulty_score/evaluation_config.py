"""Command-line and solver configuration for Numina evaluation."""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

from stp.config import Config, ProverHandlerName


REPOSITORY = Path(__file__).resolve().parents[2]
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"
EVALUATIONS_DIR = REPOSITORY / "data/evaluations"
LLM_ATTEMPTS = 32
ALPHAPROOF_ROLLOUTS = 250


def parse_args() -> argparse.Namespace:
    """Parse config and LLM inputs and return command-line arguments."""

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


def select_prover_handler(config: Config, handler: str | None) -> Config:
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
