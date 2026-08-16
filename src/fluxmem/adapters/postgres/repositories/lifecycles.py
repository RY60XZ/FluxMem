from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.memory import MemoryRow
from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.lifecycle import (
    DecisionSource,
    DecayClass,
    MemoryLifecycle,
    Status,
    Tier,
)


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
        next_reinforcement_at=row.next_reinforcement_at,
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
                next_reinforcement_at=lifecycle.next_reinforcement_at,
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
