"""Dataset loading and normalization."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from stp.config import ProverHandlerName
from stp.prover_models.registry import training_text
from stp.records import Statement, TrainingExample
from stp.storage import read_json


BY_SUFFIX = re.compile(r":=\s*by\s*$")
ASSIGN_SUFFIX = re.compile(r":=\s*$")
REPOSITORY = Path(__file__).resolve().parents[1]


def stable_id(value: str) -> str:
    """Return a short deterministic identifier for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def canonical_statement(statement: str) -> str:
    """Normalize a theorem so it ends in `:= by` without a proof."""

    statement = statement.strip()
    if "sorry" in statement:
        statement = statement.rsplit("sorry", 1)[0].strip()
    if BY_SUFFIX.search(statement):
        return BY_SUFFIX.sub(":= by", statement)
    if ASSIGN_SUFFIX.search(statement):
        return ASSIGN_SUFFIX.sub(":= by", statement)
    raise ValueError("Formal statement must end in `:=`, `:= by`, or `:= by sorry`.")


def alphaproof_theorem(statement: str) -> str:
    """Convert a canonical STP statement to AlphaProof's sorry form."""

    statement = canonical_statement(statement)
    return BY_SUFFIX.sub(":= by sorry", statement)


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL dataset records."""

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    value = read_json(path)
    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON list in {path}.")
    return value


def load_statements(config_path: Path) -> list[Statement]:
    """Load the original STP dataset configuration into canonical records."""

    datasets = read_json(config_path)
    statements = []
    for dataset in datasets:
        dataset_path = Path(dataset["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = REPOSITORY / dataset_path
        for raw in _read_records(dataset_path):
            statement = canonical_statement(raw["formal_statement"])
            labels = [str(raw.get("split", ""))]
            labels.extend(str(tag) for tag in (raw.get("tags") or []))
            labels.extend(str(tag) for tag in (dataset.get("label") or []))
            statements.append(
                Statement(
                    id=stable_id((raw.get("header") or "") + statement),
                    statement=statement,
                    header=raw.get("header"),
                    labels=tuple(label for label in labels if label),
                    matching_weight=float(dataset.get("weight", 1)),
                    source=str(dataset_path),
                )
            )
    return statements


def load_sft_examples(
    path: Path | None,
    handler: ProverHandlerName,
    tokenizer: Any,
) -> list[TrainingExample]:
    """Load SFT examples and apply the selected model format when needed."""

    if path is None:
        return []
    examples = []
    for raw in _read_records(path):
        if "prompt" in raw and "target" in raw:
            prompt = str(raw["prompt"])
            target = str(raw["target"])
            statement_id = stable_id(prompt)
        else:
            statement = canonical_statement(raw["formal_statement"])
            prompt, target = training_text(
                handler,
                statement,
                raw.get("header"),
                str(raw["proof"]).strip(),
                tokenizer,
            )
            statement_id = stable_id((raw.get("header") or "") + statement)
        examples.append(
            TrainingExample(
                prompt=prompt,
                target=target,
                weight=float(raw.get("weight", 1.0)),
                kind="sft",
                statement_id=statement_id,
                round=-1,
            )
        )
    return examples
