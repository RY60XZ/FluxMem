from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from fluxmem.application.ports.llm import (
    ModelDiagnosticsRecorder,
    ModelUsageRecorder,
)
from fluxmem.domain.lifecycle import LifecycleDecision
from fluxmem.domain.memory import Memory


class LifecycleEvaluationError(RuntimeError):
    """A lifecycle evaluator could not produce a usable semantic decision."""


@dataclass(frozen=True, slots=True)
class LifecycleEvaluationInput:
    memory: Memory
    source_role: str


class LifecycleEvaluator(Protocol):
    def evaluate_many(
        self,
        *,
        items: tuple[LifecycleEvaluationInput, ...],
        evaluated_at: datetime,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[LifecycleDecision | None, ...]: ...
