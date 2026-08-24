from __future__ import annotations

from datetime import datetime

from fluxmem.application.ports.llm import (
    ModelDiagnosticsRecorder,
    ModelUsageRecorder,
)
from fluxmem.application.ports.lifecycle import (
    LifecycleEvaluationError,
    LifecycleEvaluationInput,
    LifecycleEvaluator,
)
from fluxmem.domain.lifecycle import (
    DecisionSource,
    DecayClass,
    LifecycleDecision,
    MemoryLifecycle,
    Status,
    Tier,
)
from fluxmem.domain.memory import Memory


def tier_for_importance(importance: float) -> Tier:
    """Map semantic importance to the configured lifecycle tier."""

    if importance < 0.6:
        return Tier.WORKING
    if importance < 0.8:
        return Tier.SHORT_TERM
    return Tier.LONG_TERM


class RuleLifecycleEvaluator:
    """Conservative deterministic fallback when no valid model decision exists."""

    def evaluate_many(
        self,
        *,
        items: tuple[LifecycleEvaluationInput, ...],
        evaluated_at: datetime,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[LifecycleDecision, ...]:
        del evaluated_at, usage_recorder, diagnostics_recorder
        return tuple(
            self._decision(item=item)
            for item in items
        )

    @staticmethod
    def _decision(*, item: LifecycleEvaluationInput) -> LifecycleDecision:
        memory = item.memory
        if memory.session_applicability is not None or memory.valid_to is not None:
            importance = 0.5
            reason = "session_or_time_limited"
        else:
            importance = 0.65
            reason = "cross_session_utility"
        return LifecycleDecision(
            importance=importance,
            tier=tier_for_importance(importance),
            initial_retention=importance,
            reason_codes=(reason, f"source_role:{item.source_role}"),
            confidence=0.6,
            decision_source=DecisionSource.RULES_FALLBACK,
        )


class LifecyclePolicyExecutor:
    """Validate decisions and construct the initial lifecycle state."""

    _DECAY_CLASS_BY_TIER = {
        Tier.WORKING: DecayClass.FAST,
        Tier.SHORT_TERM: DecayClass.STANDARD,
        Tier.LONG_TERM: DecayClass.SLOW,
    }

    def initialize(
        self,
        *,
        memory: Memory,
        decision: LifecycleDecision,
        initialized_at: datetime,
    ) -> MemoryLifecycle:
        expected_tier = tier_for_importance(decision.importance)
        if decision.tier != expected_tier:
            raise ValueError(
                "lifecycle tier is inconsistent with its importance threshold"
            )

        return MemoryLifecycle(
            memory_id=memory.memory_id,
            tier=decision.tier,
            status=Status.ACTIVE,
            importance=decision.importance,
            decay_class=self._DECAY_CLASS_BY_TIER[decision.tier],
            retention_snapshot=decision.initial_retention,
            retention_anchor=initialized_at,
            reinforcement_count=0,
            use_count=0,
            last_used_at=None,
            decision_source=decision.decision_source,
            updated_at=initialized_at,
        )


class LifecycleAssigner:
    """Assign lifecycle state to a memory batch with per-item fallback."""

    def __init__(
        self,
        *,
        evaluator: LifecycleEvaluator | None = None,
        fallback: RuleLifecycleEvaluator | None = None,
        policy: LifecyclePolicyExecutor | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._fallback = fallback or RuleLifecycleEvaluator()
        self._policy = policy or LifecyclePolicyExecutor()

    def assign(
        self,
        *,
        items: tuple[LifecycleEvaluationInput, ...],
        evaluated_at: datetime,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[MemoryLifecycle, ...]:
        if not items:
            return ()
        fallback_decisions = self._fallback.evaluate_many(
            items=items,
            evaluated_at=evaluated_at,
        )
        semantic: tuple[LifecycleDecision | None, ...]
        if self._evaluator is None:
            semantic = (None,) * len(items)
        else:
            try:
                semantic = self._evaluator.evaluate_many(
                    items=items,
                    evaluated_at=evaluated_at,
                    usage_recorder=usage_recorder,
                    diagnostics_recorder=diagnostics_recorder,
                )
            except LifecycleEvaluationError:
                semantic = (None,) * len(items)
            if len(semantic) != len(items):
                semantic = (None,) * len(items)

        lifecycles: list[MemoryLifecycle] = []
        for item, proposed, fallback in zip(
            items,
            semantic,
            fallback_decisions,
        ):
            decision = proposed or fallback
            try:
                lifecycle = self._policy.initialize(
                    memory=item.memory,
                    decision=decision,
                    initialized_at=evaluated_at,
                )
            except (TypeError, ValueError):
                lifecycle = self._policy.initialize(
                    memory=item.memory,
                    decision=fallback,
                    initialized_at=evaluated_at,
                )
            lifecycles.append(lifecycle)
        return tuple(lifecycles)
