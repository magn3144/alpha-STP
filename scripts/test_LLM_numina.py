"""Try Kimina-Prover once on every Numina evaluation theorem."""

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stp.core.config import LeanSettings, load_config
from stp.data.datasets import canonical_statement
from stp.proving.lean import verify_attempts
from stp.inference.prover_models.kimina_numina import (
    TEMPERATURE,
    TOP_P,
    chat_prompt,
    extract_proof,
)
from stp.core.records import ProofRequest, SolveAttempt, Statement
from stp.data.storage import read_jsonl


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_DIR = REPOSITORY / "models/Kimina-Prover-Preview-Distill-1.5B"
DATASET_PATH = REPOSITORY / "data/dataset/numina_sft_evaluation/test.jsonl"
CONFIG_PATH = REPOSITORY / "configs/alpha_stp.toml"


def positive_int(value: str) -> int:
    """Parse one integer input and return it when it is positive."""

    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def parse_args() -> argparse.Namespace:
    """Parse model, dataset, config, and generation inputs and return them."""

    parser = argparse.ArgumentParser(
        description="Make one Kimina-Prover attempt per Numina theorem.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8196)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-problems",
        type=positive_int,
        help="Number of problems to try; defaults to the entire dataset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each theorem and its complete raw model output.",
    )
    return parser.parse_args()


def load_problems(path: Path) -> list[Statement]:
    """Load Numina JSONL records from path and return canonical statements."""

    return [
        Statement(
            id=str(value["id"]),
            statement=canonical_statement(str(value["theorem"])),
            header=None,
            labels=("numina_sft_evaluation", "test"),
            matching_weight=1.0,
            source=str(path),
        )
        for value in read_jsonl(path)
    ]


def inference_device() -> tuple[torch.device, torch.dtype]:
    """Select the available accelerator and return its device and model dtype."""

    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_model(model_path: Path) -> tuple[Any, Any, torch.device]:
    """Load a local tokenizer and model and return both plus their device."""

    device, dtype = inference_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_attempts(
    problems: list[Statement],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[list[ProofRequest], list[SolveAttempt], list[str]]:
    """Generate once per problem and return requests, attempts, and raw outputs."""

    requests = [
        ProofRequest(
            id=f"kimina-{problem.id}",
            statement_id=problem.id,
            statement=problem.statement,
            header=problem.header,
            attempt=0,
            seed=seed,
            source=problem.source,
        )
        for problem in problems
    ]
    attempts = []
    raw_outputs = []
    torch.manual_seed(seed)
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        prompts = [
            chat_prompt(request.statement, request.header, tokenizer)
            for request in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        input_length = encoded["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[:, input_length:]
        elapsed = (time.perf_counter() - started) / len(batch)
        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for request, output, tokens in zip(
            batch,
            outputs,
            generated,
            strict=True,
        ):
            raw_outputs.append(output)
            attempts.append(
                SolveAttempt(
                    request_id=request.id,
                    statement_id=request.statement_id,
                    attempt=0,
                    solver="kimina-prover-preview-distill-1.5b",
                    seed=seed,
                    status="failed",
                    proof=extract_proof(output, request.statement),
                    duration_seconds=elapsed,
                    generated_tokens=int(
                        (tokens != tokenizer.pad_token_id).sum().item()
                    ),
                    verify_seconds=0.0,
                )
            )
    return requests, attempts, raw_outputs


def verify_and_report(
    attempts: list[SolveAttempt],
    requests: list[ProofRequest],
    settings: LeanSettings,
    raw_outputs: list[str],
    verbose: bool,
) -> None:
    """Verify all attempts and print outcomes plus optional full model output."""

    verified = verify_attempts(attempts, requests, settings, ())
    solved = 0
    for index, (attempt, request, raw_output) in enumerate(
        zip(verified, requests, raw_outputs, strict=True),
        start=1,
    ):
        passed = attempt.status == "proved"
        solved += int(passed)
        result = "SOLVED" if passed else "FAILED"
        print(f"[{index}/{len(verified)}] {attempt.statement_id}: {result}")
        if verbose:
            print("Problem:")
            print(request.statement)
            print("Model output:")
            print(raw_output)
            print()
    pass_rate = 100.0 * solved / len(verified) if verified else 0.0
    print(f"Pass rate: {solved}/{len(verified)} ({pass_rate:.2f}%)")


def main() -> None:
    """Generate one proof per theorem, verify each in Lean, and report results."""

    args = parse_args()
    problems = load_problems(args.dataset)
    if args.num_problems is not None:
        problems = problems[: args.num_problems]
    model, tokenizer, device = load_model(args.model)
    requests, attempts, raw_outputs = generate_attempts(
        problems,
        model,
        tokenizer,
        args.batch_size,
        args.max_new_tokens,
        args.seed,
        device,
    )
    verify_and_report(
        attempts,
        requests,
        load_config(args.config).lean,
        raw_outputs,
        args.verbose,
    )


if __name__ == "__main__":
    main()
