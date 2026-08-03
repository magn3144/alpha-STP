"""Select and dispatch model-specific prover handlers."""

from typing import Any, Callable, Sequence

from stp.config import Config, ProverHandlerName
from stp.model import ModelRuntime
from stp.prover_models import kimina_numina, stp
from stp.prover_models.types import GeneratedProof
from stp.records import ProofRequest


ProverFunction = Callable[
    [Sequence[ProofRequest], ModelRuntime, Config],
    list[GeneratedProof],
]
TrainingTextFunction = Callable[
    [str, str | None, str, Any],
    tuple[str, str],
]

PROVER_HANDLERS: dict[ProverHandlerName, ProverFunction] = {
    "stp": stp.generate_proofs,
    "kimina_numina": kimina_numina.generate_proofs,
}
TRAINING_TEXT_HANDLERS: dict[ProverHandlerName, TrainingTextFunction] = {
    "stp": stp.training_text,
    "kimina_numina": kimina_numina.training_text,
}


def generate_proofs(
    requests: Sequence[ProofRequest],
    runtime: ModelRuntime,
    config: Config,
) -> list[GeneratedProof]:
    """Run the configured prover handler and return normalized generations."""

    return PROVER_HANDLERS[config.model.prover_handler](
        requests,
        runtime,
        config,
    )


def training_text(
    handler: ProverHandlerName,
    statement: str,
    header: str | None,
    proof: str,
    tokenizer: Any,
) -> tuple[str, str]:
    """Build model-specific training text from one verified proof."""

    return TRAINING_TEXT_HANDLERS[handler](
        statement,
        header,
        proof,
        tokenizer,
    )
