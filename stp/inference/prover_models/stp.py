"""Original STP prover prompting, generation, and answer parsing."""

from typing import Any, Sequence

from stp.core.config import Config
from stp.inference.model import ModelRuntime, generate_texts
from stp.core.records import ProofRequest
from stp.inference.prover_models.types import GeneratedProof


LEAN_CODE_PROMPT = "Complete the following Lean 4 code:\n\n```lean4\n"
PROVER_PROMPT = (
    LEAN_CODE_PROMPT
    + "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n"
)


def prover_prompt(statement: str, header: str | None) -> str:
    """Build an STP proof prompt from a statement and optional header."""

    if header is not None:
        return LEAN_CODE_PROMPT + header + statement.strip()
    return f"{PROVER_PROMPT}\n{statement.strip()}"


def extract_proof(text: str, statement: str) -> str:
    """Extract the proof suffix from one original STP completion."""

    return text.split("\n```", 1)[0]


def training_text(
    statement: str,
    header: str | None,
    proof: str,
    tokenizer: Any,
) -> tuple[str, str]:
    """Build an STP training prompt and target from a verified proof."""

    return prover_prompt(statement, header), proof


def generate_proofs(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime,
    config: Config,
) -> list[GeneratedProof]:
    """Generate STP completions and return extracted proofs with metadata."""

    outputs = generate_texts(
        runtime,
        [prover_prompt(item.statement, item.header) for item in requests],
        [item.seed for item in requests],
        config.model,
        config.solver.temperature,
        config.solver.top_p,
        max_new_tokens=config.solver.prover_max_new_tokens,
    )
    return [
        GeneratedProof(
            proof=extract_proof(text, request.statement),
            raw_output=text,
            generated_tokens=tokens,
            duration_seconds=duration,
        )
        for request, (text, tokens, duration) in zip(
            requests,
            outputs,
            strict=True,
        )
    ]
