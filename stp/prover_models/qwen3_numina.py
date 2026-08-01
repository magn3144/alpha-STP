"""Qwen3 Numina prover prompting, generation, and answer parsing."""

import re
from typing import Any, Sequence

from stp.config import Config
from stp.model import ModelRuntime, generate_texts
from stp.records import ProofRequest
from stp.prover_models.types import GeneratedProof


LEAN4_CODE_BLOCK = re.compile(r"```lean4[ \t]*\r?\n(.*?)```", re.DOTALL)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0


def proof_prompt(statement: str, header: str | None) -> str:
    """Build a Qwen3 proof request from a statement and optional header."""

    context = f"\nLean context:\n{header.strip()}\n" if header is not None else ""
    return f"""Prove the Lean 4 theorem below. Your entire response must be exactly one `lean4` Markdown code block containing the complete theorem and its proof, and nothing else. Copy the theorem statement exactly. Do not use `sorry` or `admit`.

Example question:
theorem qwen_format_example (x : ℝ) : x + 0 = x := by

Example answer:
```lean4
theorem qwen_format_example (x : ℝ) : x + 0 = x := by
  simp
```
{context}
Theorem:
{statement}
"""


def chat_prompt(
    statement: str,
    header: str | None,
    tokenizer: Any,
) -> str:
    """Apply Qwen3 chat formatting to one statement and return prompt text."""

    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert in mathematics and Lean 4. "
                    "Follow the requested proof-output format exactly."
                ),
            },
            {
                "role": "user",
                "content": proof_prompt(statement, header),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def training_text(
    statement: str,
    header: str | None,
    proof: str,
    tokenizer: Any,
) -> tuple[str, str]:
    """Build a Qwen3 chat prompt and complete fenced theorem target."""

    target = f"```lean4\n{statement}\n{proof}\n```"
    return chat_prompt(statement, header, tokenizer), target


def extract_proof(text: str, statement: str) -> str | None:
    """Extract the last exact-theorem Lean block and return its proof suffix."""

    blocks = LEAN4_CODE_BLOCK.findall(text)
    if not blocks:
        return None
    complete_proof = blocks[-1].strip()
    if not complete_proof.startswith(statement):
        return None
    proof = complete_proof[len(statement) :].strip()
    return proof or None


def generate_proofs(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime,
    config: Config,
) -> list[GeneratedProof]:
    """Generate Qwen3 completions and return extracted proofs with metadata."""

    outputs = generate_texts(
        runtime,
        [
            chat_prompt(item.statement, item.header, runtime.tokenizer)
            for item in requests
        ],
        [item.seed for item in requests],
        config.model,
        TEMPERATURE,
        TOP_P,
        max_new_tokens=config.solver.prover_max_new_tokens,
        top_k=TOP_K,
        min_p=MIN_P,
        truncate_decode=False,
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
