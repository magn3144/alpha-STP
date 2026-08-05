"""Kimina Numina prover prompting, generation, and answer parsing."""

import re
from typing import Any, Sequence

from stp.core.config import Config
from stp.inference.model import ModelRuntime, generate_texts
from stp.inference.prover_models.types import GeneratedProof
from stp.core.records import ProofRequest


LEAN4_CODE_BLOCK = re.compile(r"```lean4[ \t]*\r?\n(.*?)```", re.DOTALL)
LEAN_COMMENT = re.compile(r"/-.*?-/|--[^\r\n]*", re.DOTALL)
LEAN_DECLARATION = re.compile(
    r"\b(?:theorem|lemma)\s+.*?:=\s*by\b",
    re.DOTALL,
)
TEMPERATURE = 0.6
TOP_P = 0.95


def remove_comments(text: str) -> str:
    """Remove Lean comments from input text and return the remaining code."""

    return LEAN_COMMENT.sub(" ", text).strip()


def normalized_statement(text: str) -> str:
    """Collapse statement whitespace and return text suitable for comparison."""

    return " ".join(text.split())


def mask_comments(text: str) -> str:
    """Replace Lean comments with spaces and return code with stable offsets."""

    return LEAN_COMMENT.sub(lambda match: " " * len(match.group()), text)


def proof_prompt(statement: str, header: str | None) -> str:
    """Build a Kimina proof request from a statement and optional header."""

    context = f"\nLean context:\n{header.strip()}\n" if header is not None else ""
    statement = remove_comments(statement)
    return f"""Prove the Lean 4 theorem below. Your entire response must be exactly one `lean4` Markdown code block containing the complete theorem and its proof, and nothing else. Copy the theorem statement exactly. Do not use `sorry` or `admit`.

Example question:
theorem kimina_format_example (x : ℝ) : x + 0 = x := by

Example answer:
```lean4
theorem kimina_format_example (x : ℝ) : x + 0 = x := by
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
    """Apply the Kimina chat template and return one formatted prompt."""

    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are an expert in mathematics and Lean 4.",
            },
            {
                "role": "user",
                "content": proof_prompt(statement, header),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def training_text(
    statement: str,
    header: str | None,
    proof: str,
    tokenizer: Any,
) -> tuple[str, str]:
    """Build a Kimina chat prompt and complete fenced theorem target."""

    target = f"```lean4\n{statement}\n{proof}\n```"
    return chat_prompt(statement, header, tokenizer), target


def extract_proof(text: str, statement: str) -> str | None:
    """Match the final Lean declaration and return its proof suffix."""

    blocks = LEAN4_CODE_BLOCK.findall(text)
    if not blocks:
        return None
    complete_proof = blocks[-1].strip()
    comparable_proof = mask_comments(complete_proof)
    expected_declaration = normalized_statement(
        LEAN_DECLARATION.findall(remove_comments(statement))[0]
    )
    for declaration in LEAN_DECLARATION.finditer(comparable_proof):
        if normalized_statement(declaration.group()) == expected_declaration:
            proof = complete_proof[declaration.end() :].strip()
            return proof or None
    return None


def generate_proofs(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime,
    config: Config,
) -> list[GeneratedProof]:
    """Generate Kimina completions and return proofs with generation metadata."""

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
