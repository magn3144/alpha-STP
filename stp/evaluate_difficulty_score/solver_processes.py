"""Isolated GPU solver processes for Numina evaluation."""

from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from stp.core.config import Config
from stp.inference.model import load_runtime
from stp.core.records import ProofRequest, SolveAttempt
from stp.proving.solvers import solve_with_alphaproof, solve_with_llm
from stp.data.storage import load_records, read_jsonl, write_jsonl


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
            raise RuntimeError(f"LLM worker exited with status {exitcode}.")
        return load_records(output_path, SolveAttempt)


def solve_with_alphaproof_process(
    requests: Sequence[ProofRequest],
    config: Config,
) -> tuple[list[SolveAttempt], dict[str, Any]]:
    """Run one AlphaProof subprocess and return normalized attempts and raw result."""

    with TemporaryDirectory(prefix="alpha-stp-evaluation-") as temporary:
        artifact_dir = Path(temporary)
        attempts = solve_with_alphaproof(requests, config, artifact_dir)
        raw_result = read_jsonl(artifact_dir / "alphaproof_results.jsonl")[0]
    return attempts, raw_result
