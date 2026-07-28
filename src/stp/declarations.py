import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from stp.config import Config
from stp.records import TheoremDeclaration
from stp.storage import read_json, read_jsonl, write_json, write_jsonl


def _git_revision(path: Path) -> str:
    """Return the Git revision containing the given path."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def declaration_manifest_path(path: Path) -> Path:
    """Return the manifest path paired with a declaration JSONL artifact."""

    return path.with_suffix(path.suffix + ".manifest.json")


def declaration_provenance(config: Config) -> dict[str, Any]:
    """Describe the Lean environment used by the declaration artifact."""

    mathlib = config.lean.project_dir / ".lake" / "packages" / "mathlib"
    repository = Path(__file__).resolve().parents[2]
    return {
        "lean_toolchain": (
            config.lean.project_dir / "lean-toolchain"
        ).read_text(encoding="utf-8").strip(),
        "imports": list(config.lean.imports),
        "lean_project_revision": _git_revision(config.lean.project_dir),
        "mathlib_revision": _git_revision(mathlib),
        "generator_revision": _git_revision(repository),
    }


def _lean_declaration_program(imports: Sequence[str], output: Path) -> str:
    """Build a Lean program that writes imported theorem declarations."""

    path = json.dumps(str(output))
    import_lines = "\n".join(f"import {module}" for module in imports)
    return f"""{import_lines}

open Lean Elab Command Meta

run_cmd liftTermElabM do
  let env ← getEnv
  let mut rows := #[]
  for (name, info) in env.constants.toList do
    match info with
    | .thmInfo value =>
      if !name.isInternal then
        let type ← ppExpr value.type
        let row := Lean.Json.mkObj [
          ("full_name", Lean.Json.str name.toString),
          ("statement", Lean.Json.str s!"theorem {{name}} : {{type}}")
        ]
        rows := rows.push row.compress
    | _ => pure ()
  IO.FS.writeFile {path} ("\\n".intercalate rows.toList ++ "\\n")
"""


def build_declaration_artifact(config: Config) -> dict[str, Any]:
    """Generate the configured theorem-declaration JSONL and manifest."""

    output = config.data.theorem_declarations
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stp-declarations-") as directory:
        temporary_dir = Path(directory)
        lean_path = temporary_dir / "declarations.lean"
        raw_path = temporary_dir / "declarations.jsonl"
        lean_path.write_text(
            _lean_declaration_program(config.lean.imports, raw_path),
            encoding="utf-8",
        )
        subprocess.run(
            ["lake", "env", "lean", str(lean_path)],
            cwd=config.lean.project_dir,
            check=True,
        )
        values = sorted(
            read_jsonl(raw_path),
            key=lambda value: str(value["full_name"]),
        )
        declarations = [
            TheoremDeclaration(
                full_name=str(value["full_name"]),
                statement=str(value["statement"]),
            )
            for value in values
        ]
    write_jsonl(output, declarations)
    manifest = declaration_provenance(config) | {
        "artifact": str(output),
        "declarations": len(declarations),
    }
    write_json(declaration_manifest_path(output), manifest)
    return manifest


def validate_declaration_artifact(config: Config) -> None:
    """Fail when declarations do not match the configured Lean environment."""

    path = config.data.theorem_declarations
    manifest = read_json(declaration_manifest_path(path))
    expected = declaration_provenance(config)
    for key in (
        "lean_toolchain",
        "imports",
        "lean_project_revision",
        "mathlib_revision",
    ):
        if manifest[key] != expected[key]:
            raise ValueError(
                f"Theorem declaration artifact has stale {key}: {path}"
            )


def load_declaration_map(config: Config) -> dict[str, str]:
    """Load and validate theorem declarations by fully qualified name."""

    validate_declaration_artifact(config)
    declarations = {}
    with config.data.theorem_declarations.open(encoding="utf-8") as file:
        for line in file:
            value = json.loads(line)
            declarations[str(value["full_name"])] = str(value["statement"])
    return declarations


def load_declaration_names(config: Config) -> frozenset[str]:
    """Load and validate only the theorem names needed for verification."""

    validate_declaration_artifact(config)
    names = set()
    with config.data.theorem_declarations.open(encoding="utf-8") as file:
        for line in file:
            names.add(str(json.loads(line)["full_name"]))
    return frozenset(names)
