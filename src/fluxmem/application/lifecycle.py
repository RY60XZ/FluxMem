from __future__ import annotations

from datetime import datetime

from fluxmem.domain.lifecycle import (
    DecisionSource,
    DecayClass,
    LifecycleDecision,
    MemoryLifecycle,
    Status,
    Tier,
)
from fluxmem.domain.memory import Memory


class RuleLifecycleEvaluator:
    """Conservative deterministic fallback when no valid model decision exists."""

    def evaluate(
        self,
        *,
        memory: Memory,
        source_role: str,
        evaluated_at: datetime,
    ) -> LifecycleDecision:
        del evaluated_at

        if memory.session_applicability is not None or memory.valid_to is not None:
            importance = 0.5
            tier = Tier.WORKING
            reason = "session_or_time_limited"
        else:
            importance = 0.65
            tier = Tier.SHORT_TERM
            reason = "cross_session_utility"

        return LifecycleDecision(
            importance=importance,
            tier=tier,
            initial_retention=importance,
            reason_codes=(reason, f"source_role:{source_role}"),
            confidence=0.6,
            decision_source=DecisionSource.RULES_FALLBACK,
        )


class LifecyclePolicyExecutor:
    """Validate decisions and produce the only valid initial lifecycle state."""

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
        expected_tier = self._tier_for_importance(decision.importance)
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
            next_reinforcement_at=None,
            decision_source=decision.decision_source,
            updated_at=initialized_at,
        )

    @staticmethod
    def _tier_for_importance(importance: float) -> Tier:
        if importance < 0.6:
            return Tier.WORKING
        if importance < 0.8:
            return Tier.SHORT_TERM
        return Tier.LONG_TERM
