from __future__ import annotations

from threading import Lock
from typing import Protocol

from fluxmem.domain.llm import LLMUsageReport, ModelCallUsage


class _UsageRecorder(Protocol):
    def record(self, call: ModelCallUsage) -> None: ...


class ModelUsageCollector:
    """Thread-safe, turn-local collector for model-call usage metadata."""

    def __init__(self, *, forward_to: _UsageRecorder | None = None) -> None:
        self._calls: list[ModelCallUsage] = []
        self._lock = Lock()
        self._forward_to = forward_to

    def record(self, call: ModelCallUsage) -> None:
        with self._lock:
            self._calls.append(call)
        if self._forward_to is not None:
            self._forward_to.record(call)

    def snapshot(self) -> LLMUsageReport:
        with self._lock:
            return LLMUsageReport(calls=tuple(self._calls))
