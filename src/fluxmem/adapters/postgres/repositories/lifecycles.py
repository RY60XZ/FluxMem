from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.memory import MemoryRow
from fluxmem.adapters.postgres.models.memory_usage import MemoryUsageRow
from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.info_pack import MemoryUsage, UsageType
from fluxmem.domain.lifecycle import (
    DecisionSource,
    DecayClass,
    MemoryLifecycle,
    Status,
    Tier,
)


_REINFORCEMENT_RATE = {
    UsageType.CONTEXT_INCLUDED: 0.1,
    UsageType.MODEL_ATTRIBUTED: 0.2,
}
_DECAY_CLASS_BY_TIER = {
    Tier.WORKING: DecayClass.FAST,
    Tier.SHORT_TERM: DecayClass.STANDARD,
    Tier.LONG_TERM: DecayClass.SLOW,
}
_SHORT_TERM_MIN_USAGE_DAYS = 2

def _to_domain(row: MemoryLifecycleRow) -> MemoryLifecycle:
    return MemoryLifecycle(
        memory_id=row.memory_id,
        tier=Tier(row.tier),
        status=Status(row.status),
        importance=row.importance,
        decay_class=DecayClass(row.decay_class),
        retention_snapshot=row.retention_snapshot,
        retention_anchor=row.retention_anchor,
        reinforcement_count=row.reinforcement_count,
        use_count=row.use_count,
        last_used_at=row.last_used_at,
        decision_source=DecisionSource(row.decision_source),
        updated_at=row.updated_at,
    )


class SqlAlchemyLifecycleRepository:
    """Persist lifecycle state in the memory's transaction."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, lifecycle: MemoryLifecycle) -> None:
        self._session.add(
            MemoryLifecycleRow(
                memory_id=lifecycle.memory_id,
                tier=lifecycle.tier.value,
                status=lifecycle.status.value,
                importance=lifecycle.importance,
                decay_class=lifecycle.decay_class.value,
                retention_snapshot=lifecycle.retention_snapshot,
                retention_anchor=lifecycle.retention_anchor,
                reinforcement_count=lifecycle.reinforcement_count,
                use_count=lifecycle.use_count,
                last_used_at=lifecycle.last_used_at,
                decision_source=lifecycle.decision_source.value,
                updated_at=lifecycle.updated_at,
            )
        )

    def get(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
    ) -> MemoryLifecycle | None:
        statement = (
            select(MemoryLifecycleRow)
            .join(MemoryRow, MemoryRow.memory_id == MemoryLifecycleRow.memory_id)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                MemoryLifecycleRow.memory_id == memory_id,
                SessionRow.user_id == user_id,
            )
        )
        row = self._session.scalar(statement)
        if row is None:
            return None
        return _to_domain(row)

    def reinforce(
        self,
        *,
        memory_id: UUID,
        query_id: UUID,
        user_id: UUID,
        session_id: UUID,
        usage: MemoryUsage,
        used_at: datetime,
    ) -> MemoryLifecycle | None:
        """Record and reinforce one unique actual use under a row lock."""

        statement = (
            select(MemoryLifecycleRow)
            .join(MemoryRow, MemoryRow.memory_id == MemoryLifecycleRow.memory_id)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                MemoryLifecycleRow.memory_id == memory_id,
                MemoryLifecycleRow.status == Status.ACTIVE.value,
                SessionRow.user_id == user_id,
                or_(
                    MemoryRow.session_applicability.is_(None),
                    MemoryRow.session_applicability == session_id,
                ),
            )
            .with_for_update(of=MemoryLifecycleRow)
        )
        row = self._session.scalar(statement)
        if row is None:
            return None

        lifecycle = _to_domain(row)
        recorded_usage = self._session.scalar(
            select(MemoryUsageRow).where(
                MemoryUsageRow.memory_id == memory_id,
                MemoryUsageRow.query_id == query_id,
            )
        )
        if recorded_usage is not None:
            _retain_strongest_usage(recorded_usage, usage)
            return lifecycle

        self._session.add(
            MemoryUsageRow(
                memory_id=memory_id,
                query_id=query_id,
                usage_type=usage.usage_type.value,
                rank=usage.rank,
                contribution=usage.contribution,
                used_at=used_at,
            )
        )
        self._session.flush()

        row.use_count += 1
        row.last_used_at = used_at
        row.updated_at = used_at

        rate = _REINFORCEMENT_RATE[usage.usage_type]
        current_retention = lifecycle.retention_at(used_at)
        row.retention_snapshot = min(
            1.0,
            current_retention + rate * (1.0 - current_retention),
        )
        row.retention_anchor = used_at
        row.importance = min(
            1.0,
            lifecycle.importance + rate * (1.0 - lifecycle.importance),
        )
        row.reinforcement_count += 1

        usage_days = len(
            {
                _utc_date(recorded_at)
                for recorded_at in self._session.scalars(
                    select(MemoryUsageRow.used_at).where(
                        MemoryUsageRow.memory_id == memory_id
                    )
                )
            }
        )
        tier = _promoted_tier(
            current_tier=lifecycle.tier,
            importance=row.importance,
            usage_days=usage_days,
        )
        if tier != lifecycle.tier:
            row.tier = tier.value
            row.decay_class = _DECAY_CLASS_BY_TIER[tier].value

        return _to_domain(row)


def _retain_strongest_usage(row: MemoryUsageRow, usage: MemoryUsage) -> None:
    """Keep the strongest signal without applying reinforcement twice."""

    priority = {
        UsageType.CONTEXT_INCLUDED: 0,
        UsageType.MODEL_ATTRIBUTED: 1,
    }
    recorded_type = UsageType(row.usage_type)
    if priority[usage.usage_type] > priority[recorded_type]:
        row.usage_type = usage.usage_type.value
        row.rank = usage.rank
        row.contribution = usage.contribution
        return

    if usage.usage_type != recorded_type:
        return

    if usage.rank is not None and (row.rank is None or usage.rank < row.rank):
        row.rank = usage.rank
    if usage.contribution is not None and (
        row.contribution is None or usage.contribution > row.contribution
    ):
        row.contribution = usage.contribution


def _promoted_tier(
    *,
    current_tier: Tier,
    importance: float,
    usage_days: int,
) -> Tier:
    """Promote at most one tier after use on enough distinct UTC dates."""

    if (
        current_tier == Tier.WORKING
        and importance >= 0.6
        and usage_days >= _SHORT_TERM_MIN_USAGE_DAYS
    ):
        return Tier.SHORT_TERM
    if (
        current_tier == Tier.SHORT_TERM
        and importance >= 0.8
    ):
        return Tier.LONG_TERM
    return current_tier


def _utc_date(value: datetime) -> date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(timezone.utc).date()
