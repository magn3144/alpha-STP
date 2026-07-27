import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


@dataclass(frozen=True)
class RunSettings:
    """STP curriculum and output settings."""

    output_dir: Path
    rounds: int
    statements_per_round: int
    replay_rounds: int
    seed: int
    conjecture_multiplier: int
    conjecture_threshold: float
    training_proof_threshold: float


@dataclass(frozen=True)
class ModelSettings:
    """Causal model inference and full-training settings."""

    name: str
    tokenizer: str
    max_sequence_length: int
    max_new_tokens: int
    generation_batch_size: int
    embedding_batch_size: int
    train_microbatch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_checkpointing: bool


@dataclass(frozen=True)
class SolverSettings:
    """LLM or AlphaProof solver settings."""

    kind: Literal["llm", "alphaproof"]
    attempts_per_statement: int
    temperature: float
    top_p: float
    alphaproof_command: tuple[str, ...]
    alphaproof_run_dir: Path
    alphaproof_num_simulations: int
    alphaproof_num_sampled_actions: int
    alphaproof_timeout_seconds: float


@dataclass(frozen=True)
class LeanSettings:
    """LeanTree verification settings."""

    project_dir: Path
    imports: tuple[str, ...]
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class DataSettings:
    """Input dataset and theorem dictionary paths."""

    dataset_config: Path
    sft_dataset: Path | None


@dataclass(frozen=True)
class Config:
    """Complete immutable configuration for an STP run."""

    path: Path
    run: RunSettings
    model: ModelSettings
    solver: SolverSettings
    lean: LeanSettings
    data: DataSettings


def _path(base: Path, value: str) -> Path:
    """Resolve a configuration path relative to the TOML file."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> Config:
    """Read TOML configuration and return typed settings."""

    path = path.resolve()
    base = path.parent
    with path.open("rb") as file:
        data: dict[str, Any] = tomllib.load(file)

    run = data["run"]
    model = data["model"]
    solver = data["solver"]
    lean = data["lean"]
    inputs = data["data"]
    return Config(
        path=path,
        run=RunSettings(
            output_dir=_path(base, run["output_dir"]),
            rounds=int(run["rounds"]),
            statements_per_round=int(run["statements_per_round"]),
            replay_rounds=int(run["replay_rounds"]),
            seed=int(run["seed"]),
            conjecture_multiplier=int(run["conjecture_multiplier"]),
            conjecture_threshold=float(run["conjecture_threshold"]),
            training_proof_threshold=float(run["training_proof_threshold"]),
        ),
        model=ModelSettings(
            name=str(model["name"]),
            tokenizer=str(model["tokenizer"]),
            max_sequence_length=int(model["max_sequence_length"]),
            max_new_tokens=int(model["max_new_tokens"]),
            generation_batch_size=int(model["generation_batch_size"]),
            embedding_batch_size=int(model["embedding_batch_size"]),
            train_microbatch_size=int(model["train_microbatch_size"]),
            gradient_accumulation_steps=int(model["gradient_accumulation_steps"]),
            epochs=int(model["epochs"]),
            learning_rate=float(model["learning_rate"]),
            weight_decay=float(model["weight_decay"]),
            warmup_steps=int(model["warmup_steps"]),
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        ),
        solver=SolverSettings(
            kind=cast(Literal["llm", "alphaproof"], solver["kind"]),
            attempts_per_statement=int(solver["attempts_per_statement"]),
            temperature=float(solver["temperature"]),
            top_p=float(solver["top_p"]),
            alphaproof_command=tuple(solver["alphaproof_command"]),
            alphaproof_run_dir=_path(base, solver["alphaproof_run_dir"]),
            alphaproof_num_simulations=int(
                solver["alphaproof_num_simulations"]
            ),
            alphaproof_num_sampled_actions=int(
                solver["alphaproof_num_sampled_actions"]
            ),
            alphaproof_timeout_seconds=float(
                solver["alphaproof_timeout_seconds"]
            ),
        ),
        lean=LeanSettings(
            project_dir=_path(base, lean["project_dir"]),
            imports=tuple(lean["imports"]),
            workers=int(lean["workers"]),
            timeout_seconds=float(lean["timeout_seconds"]),
        ),
        data=DataSettings(
            dataset_config=_path(base, inputs["dataset_config"]),
            sft_dataset=(
                _path(base, inputs["sft_dataset"])
                if "sft_dataset" in inputs
                else None
            ),
        ),
    )
