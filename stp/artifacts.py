"""Round artifact paths, history loading, and provenance."""

import shutil
import subprocess
from pathlib import Path

from stp.config import Config
from stp.declarations import (
    declaration_manifest_path,
    validate_declaration_artifact,
)
from stp.records import (
    Conjecture,
    ConjectureAssessment,
    SolveAttempt,
    Statement,
)
from stp.storage import load_records, read_json, write_json


def round_dir(config: Config, round_index: int) -> Path:
    """Return one round's artifact directory."""

    return config.run.output_dir / f"round_{round_index:03d}"


def input_checkpoint(config: Config, round_index: int) -> str | Path:
    """Return the checkpoint used at the start of a round."""

    if round_index == 0:
        return config.model.name
    return round_dir(config, round_index - 1) / "checkpoint"


def load_round_pool(directory: Path) -> list[Statement | Conjecture]:
    """Load selected statements and generated conjectures."""

    statements = load_records(directory / "selected.jsonl", Statement)
    conjectures = load_records(directory / "conjectures.jsonl", Conjecture)
    return [*statements, *conjectures]


def previous_scores(
    config: Config,
    round_index: int,
) -> list[ConjectureAssessment]:
    """Return maximum historical solve rates."""

    best: dict[str, ConjectureAssessment] = {}
    for previous in range(round_index):
        path = round_dir(config, previous) / "assessments.jsonl"
        if not path.exists():
            continue
        for score in load_records(path, ConjectureAssessment):
            old = best.get(score.statement_id)
            if old is None or score.score > old.score:
                best[score.statement_id] = score
    return list(best.values())


def historical_attempts(
    config: Config,
    round_index: int,
) -> list[SolveAttempt]:
    """Load attempts from completed earlier rounds."""

    attempts = []
    for previous in range(round_index):
        path = round_dir(config, previous) / "solve_attempts.jsonl"
        if path.exists():
            attempts.extend(load_records(path, SolveAttempt))
    return attempts


def _git_revision(repository: Path) -> dict[str, str | bool]:
    """Read a repository's commit and dirty status."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status)}


def write_manifest(config: Config) -> None:
    """Record code, Lean, model, and configuration provenance."""

    path = config.run.output_dir / "manifest.json"
    validate_declaration_artifact(config)
    if path.exists():
        manifest = read_json(path)
        if manifest.get("artifact_schema") != 2:
            raise ValueError(
                "Run artifacts use an unsupported schema. Start a new run."
            )
        return

    repository = Path(__file__).resolve().parents[1]
    alphaproof = repository.parent / "AlphaProof"
    lean_version = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=config.lean.project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        path,
        {
            "artifact_schema": 2,
            "stp": _git_revision(repository),
            "alphaproof": _git_revision(alphaproof),
            "leantree": _git_revision(alphaproof / "vendor/leantree"),
            "lean_version": lean_version,
            "lean_project": str(config.lean.project_dir),
            "base_model": config.model.name,
            "solver": config.solver.kind,
            "theorem_declarations": read_json(
                declaration_manifest_path(
                    config.data.theorem_declarations
                )
            ),
        },
    )
    shutil.copy2(config.path, config.run.output_dir / "config.toml")
