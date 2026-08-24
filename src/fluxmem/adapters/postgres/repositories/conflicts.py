from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.conflict import MemoryConflictRow
from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.memory import MemoryRow
from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.conflict import ConflictNeighbor, MemoryConflict
from fluxmem.domain.lifecycle import (
    RETENTION_FLOOR,
    STABILITY_DAYS,
    DecayClass,
    Status,
)
from fluxmem.domain.memory import Memory


def _memory_to_domain(row: MemoryRow) -> Memory:
    return Memory(
        memory_id=row.memory_id,
        message_id=row.message_id,
        content=row.content,
        created_at=row.created_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        session_applicability=row.session_applicability,
    )


def _conflict_to_domain(row: MemoryConflictRow) -> MemoryConflict:
    return MemoryConflict(
        memory_a_id=row.memory_a_id,
        memory_b_id=row.memory_b_id,
        confidence=row.confidence,
        created_at=row.created_at,
    )


class SqlAlchemyConflictRepository:
    """Persist canonical edges and hydrate eligible one-hop neighbors."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, conflict: MemoryConflict) -> None:
        statement = (
            insert(MemoryConflictRow)
            .values(
                memory_a_id=conflict.memory_a_id,
                memory_b_id=conflict.memory_b_id,
                confidence=conflict.confidence,
                created_at=conflict.created_at,
            )
            .on_conflict_do_nothing(
                index_elements=(
                    MemoryConflictRow.memory_a_id,
                    MemoryConflictRow.memory_b_id,
                )
            )
        )
        self._session.execute(statement)

    def expand(
        self,
        *,
        seed_memory_ids: tuple[UUID, ...],
        user_id: UUID,
        session_id: UUID,
        as_of: datetime,
    ) -> tuple[ConflictNeighbor, ...]:
        if not seed_memory_ids:
            return ()

        seed_id = case(
            (
                MemoryConflictRow.memory_a_id.in_(seed_memory_ids),
                MemoryConflictRow.memory_a_id,
            ),
            else_=MemoryConflictRow.memory_b_id,
        )
        neighbor_id = case(
            (
                MemoryConflictRow.memory_a_id.in_(seed_memory_ids),
                MemoryConflictRow.memory_b_id,
            ),
            else_=MemoryConflictRow.memory_a_id,
        )
        elapsed_days = func.greatest(
            0.0,
            func.extract(
                "epoch",
                literal(as_of) - MemoryLifecycleRow.retention_anchor,
            )
            / 86_400.0,
        )
        stability_days = case(
            (
                MemoryLifecycleRow.decay_class == DecayClass.FAST.value,
                STABILITY_DAYS[DecayClass.FAST],
            ),
            (
                MemoryLifecycleRow.decay_class == DecayClass.STANDARD.value,
                STABILITY_DAYS[DecayClass.STANDARD],
            ),
            else_=STABILITY_DAYS[DecayClass.SLOW],
        )
        retention = (
            RETENTION_FLOOR
            + (MemoryLifecycleRow.retention_snapshot - RETENTION_FLOOR)
            * func.exp(-elapsed_days / stability_days)
        ).label("retention")
        statement = (
            select(
                MemoryConflictRow,
                seed_id.label("seed_id"),
                MemoryRow,
                retention,
            )
            .select_from(MemoryConflictRow)
            .join(MemoryRow, MemoryRow.memory_id == neighbor_id)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .join(
                MemoryLifecycleRow,
                MemoryLifecycleRow.memory_id == MemoryRow.memory_id,
            )
            .where(
                or_(
                    MemoryConflictRow.memory_a_id.in_(seed_memory_ids),
                    MemoryConflictRow.memory_b_id.in_(seed_memory_ids),
                ),
                SessionRow.user_id == user_id,
                MemoryLifecycleRow.status == Status.ACTIVE.value,
                or_(
                    MemoryRow.session_applicability.is_(None),
                    MemoryRow.session_applicability == session_id,
                ),
            )
            .order_by(
                seed_id,
                MemoryConflictRow.confidence.desc().nullslast(),
                neighbor_id,
            )
        )
        rows = self._session.execute(statement).all()
        return tuple(
            ConflictNeighbor(
                seed_memory_id=row.seed_id,
                memory=_memory_to_domain(row[2]),
                conflict=_conflict_to_domain(row[0]),
                retention=float(row.retention),
            )
            for row in rows
        )
