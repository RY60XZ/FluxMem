from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import exp
from uuid import UUID


RETENTION_FLOOR = 0.1
RELEVANCE_FLOOR = 0.6


class Tier(StrEnum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class Status(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"


class DecayClass(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    SLOW = "slow"


class DecisionSource(StrEnum):
    LLM_PRIMARY = "llm_primary"
    LLM_RETRY = "llm_retry"
    RULES_FALLBACK = "rules_fallback"
    MANUAL = "manual"


STABILITY_DAYS: dict[DecayClass, float] = {
    DecayClass.FAST: 1.5,
    DecayClass.STANDARD: 10.5,
    DecayClass.SLOW: 90.0,
}


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    """A provider-independent initial lifecycle judgment."""

    importance: float
    tier: Tier
    initial_retention: float
    reason_codes: tuple[str, ...]
    confidence: float
    decision_source: DecisionSource

    def __post_init__(self) -> None:
        if not RETENTION_FLOOR <= self.importance <= 1.0:
            raise ValueError(
                f"importance must be between {RETENTION_FLOOR} and 1"
            )
        if self.initial_retention != self.importance:
            raise ValueError("initial retention must equal importance")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason_codes:
            raise ValueError("at least one lifecycle reason code is required")


@dataclass(frozen=True, slots=True)
class MemoryLifecycle:
    """Mutable lifecycle state represented as an immutable application value."""

    memory_id: UUID
    tier: Tier
    status: Status
    importance: float
    decay_class: DecayClass
    retention_snapshot: float
    retention_anchor: datetime
    reinforcement_count: int
    use_count: int
    last_used_at: datetime | None
    next_reinforcement_at: datetime | None
    decision_source: DecisionSource
    updated_at: datetime

    def __post_init__(self) -> None:
        if not RETENTION_FLOOR <= self.importance <= 1.0:
            raise ValueError(
                f"importance must be between {RETENTION_FLOOR} and 1"
            )
        if not RETENTION_FLOOR <= self.retention_snapshot <= 1.0:
            raise ValueError(
                f"retention snapshot must be between {RETENTION_FLOOR} and 1"
            )
        if self.reinforcement_count < 0 or self.use_count < 0:
            raise ValueError("lifecycle counters cannot be negative")

    def retention_at(self, as_of: datetime) -> float:
        """Calculate decay lazily"""

        elapsed_days = max(
            0.0,
            (as_of - self.retention_anchor).total_seconds() / 86_400.0,
        )
        stability = STABILITY_DAYS[self.decay_class]
        return RETENTION_FLOOR + (
            self.retention_snapshot - RETENTION_FLOOR
        ) * exp(-elapsed_days / stability)

    def adjust_relevance(self, base_score: float, as_of: datetime) -> float:
        retention = self.retention_at(as_of)
        return base_score * (
            RELEVANCE_FLOOR + (1.0 - RELEVANCE_FLOOR) * retention
        )
