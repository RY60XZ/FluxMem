from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fluxmem.application.ports.llm import (
    ModelDiagnosticsRecorder,
    ModelUsageRecorder,
)
from fluxmem.domain.lifecycle import LifecycleDecision
from fluxmem.domain.memory import Memory


class LifecycleEvaluator(Protocol):
    def evaluate(
        self,
        *,
        memory: Memory,
        source_role: str,
        evaluated_at: datetime,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> LifecycleDecision: ...
