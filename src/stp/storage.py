import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast

from stp.records import record_to_dict


Record = TypeVar("Record")


def read_json(path: Path) -> Any:
    """Read one JSON value from a local file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    """Atomically write one JSON value to a local file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects from a JSONL file."""

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: Iterable[object]) -> None:
    """Atomically write dataclasses or mappings as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            value = record if isinstance(record, dict) else record_to_dict(record)
            file.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_records(path: Path, record_type: type[Record]) -> list[Record]:
    """Load JSONL objects into a dataclass record type."""

    names = {item.name for item in fields(cast(Any, record_type))}
    records = []
    for value in read_jsonl(path):
        values = {key: item for key, item in value.items() if key in names}
        for name in (
            "labels",
            "invoked_lemmas",
            "alphaproof_command",
            "imports",
        ):
            if name in values:
                values[name] = tuple(values[name])
        records.append(record_type(**values))
    return records
