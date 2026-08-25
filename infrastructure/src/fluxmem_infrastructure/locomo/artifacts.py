from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class ArtifactWriter:
    """Write or resume one benchmark run through atomic JSON artifacts."""

    def __init__(self, output_dir: Path, *, resume: bool = False) -> None:
        if resume:
            if not output_dir.is_dir():
                raise ValueError(
                    f"benchmark output does not exist: {output_dir}"
                )
            if not (output_dir / "manifest.json").is_file():
                raise ValueError(
                    f"benchmark manifest does not exist: {output_dir}"
                )
        elif output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"benchmark output is not empty: {output_dir}")
        self.output_dir = output_dir
        self.conversation_dir = output_dir / "conversations"
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, value: Mapping[str, Any]) -> None:
        _write_json(self.output_dir / "manifest.json", value)

    def read_manifest(self) -> dict[str, Any]:
        return _read_json(self.output_dir / "manifest.json")

    def write_checkpoint(self, value: Mapping[str, Any]) -> None:
        _write_json(self.output_dir / "checkpoint.json", value)

    def read_checkpoint(self) -> dict[str, Any]:
        return _read_json(self.output_dir / "checkpoint.json")

    def write_conversation(
        self,
        *,
        sample_id: str,
        value: Mapping[str, Any],
    ) -> None:
        if not sample_id or any(character in sample_id for character in "/\\"):
            raise ValueError("sample_id is unsafe for an artifact filename")
        _write_json(self.conversation_dir / f"{sample_id}.json", value)

    def write_summary(self, value: Mapping[str, Any]) -> None:
        _write_json(self.output_dir / "summary.json", value)

    def read_conversation(self, *, sample_id: str) -> dict[str, Any]:
        if not sample_id or any(character in sample_id for character in "/\\"):
            raise ValueError("sample_id is unsafe for an artifact filename")
        return _read_json(self.conversation_dir / f"{sample_id}.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"benchmark artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid benchmark artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"benchmark artifact must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
