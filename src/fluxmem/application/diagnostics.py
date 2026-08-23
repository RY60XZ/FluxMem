from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Mapping
from uuid import UUID

from fluxmem.domain.info_pack import TurnMemoryPacks
from fluxmem.domain.llm import (
    ConversationTurnDiagnostics,
    MemoryWriteOutcome,
    ModelCallDiagnostics,
)


class TurnDiagnosticsCollector:
    """Thread-safe mutable collector behind immutable trace snapshots."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._retrieval: TurnMemoryPacks | None = None
        self._answer_context_memory_ids: tuple[UUID, ...] = ()
        self._model_calls: list[ModelCallDiagnostics] = []
        self._feedback_applied: bool | None = None
        self._memory_outcomes: tuple[MemoryWriteOutcome, ...] = ()
        self._post_answer_errors: tuple[str, ...] = ()
        self._complete = False

    def record_retrieval(self, retrieval: TurnMemoryPacks) -> None:
        with self._lock:
            self._retrieval = retrieval

    def record_answer_context(self, memory_ids: tuple[UUID, ...]) -> None:
        with self._lock:
            self._answer_context_memory_ids = memory_ids

    def record(self, call: ModelCallDiagnostics) -> None:
        with self._lock:
            self._model_calls.append(call)

    def complete(
        self,
        *,
        feedback_applied: bool,
        memory_outcomes: tuple[MemoryWriteOutcome, ...],
        post_answer_errors: tuple[str, ...],
    ) -> None:
        with self._lock:
            self._feedback_applied = feedback_applied
            self._memory_outcomes = memory_outcomes
            self._post_answer_errors = post_answer_errors
            self._complete = True

    def snapshot(self) -> ConversationTurnDiagnostics:
        with self._lock:
            return ConversationTurnDiagnostics(
                retrieval=self._retrieval,
                answer_context_memory_ids=self._answer_context_memory_ids,
                model_calls=tuple(self._model_calls),
                feedback_applied=self._feedback_applied,
                memory_outcomes=self._memory_outcomes,
                post_answer_errors=self._post_answer_errors,
                complete=self._complete,
            )


def diagnostics_to_dict(
    diagnostics: ConversationTurnDiagnostics,
) -> dict[str, Any]:
    """Convert a trace to JSON-compatible public data."""

    converted = _json_value(asdict(diagnostics))
    if not isinstance(converted, dict):
        raise TypeError("conversation diagnostics must serialize to an object")
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
