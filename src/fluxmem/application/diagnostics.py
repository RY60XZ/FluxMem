from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Mapping
from uuid import UUID

from fluxmem.domain.info_pack import MemoryPack
from fluxmem.domain.llm import (
    MemoryDiagnostics,
    MemoryWriteOutcome,
    ModelCallDiagnostics,
)


class MemoryDiagnosticsCollector:
    """Thread-safe mutable collector behind immutable trace snapshots."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._retrieval: MemoryPack | None = None
        self._model_calls: list[ModelCallDiagnostics] = []
        self._memory_outcomes: tuple[MemoryWriteOutcome, ...] = ()
        self._errors: tuple[str, ...] = ()
        self._complete = False

    def record_retrieval(self, retrieval: MemoryPack) -> None:
        with self._lock:
            self._retrieval = retrieval

    def record(self, call: ModelCallDiagnostics) -> None:
        with self._lock:
            self._model_calls.append(call)

    def complete(
        self,
        *,
        memory_outcomes: tuple[MemoryWriteOutcome, ...],
        errors: tuple[str, ...],
    ) -> None:
        with self._lock:
            self._memory_outcomes = memory_outcomes
            self._errors = errors
            self._complete = True

    def snapshot(self) -> MemoryDiagnostics:
        with self._lock:
            return MemoryDiagnostics(
                retrieval=self._retrieval,
                model_calls=tuple(self._model_calls),
                memory_outcomes=self._memory_outcomes,
                errors=self._errors,
                complete=self._complete,
            )


def diagnostics_to_dict(
    diagnostics: MemoryDiagnostics,
) -> dict[str, Any]:
    """Convert a trace to JSON-compatible public data."""

    converted = _json_value(asdict(diagnostics))
    if not isinstance(converted, dict):
        raise TypeError("memory diagnostics must serialize to an object")
    return converted


def _json_value(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
