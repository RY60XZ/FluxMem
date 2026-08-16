from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.memory import MemoryRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.memory import Memory


class SqlAlchemyMemoryRepository:
    """Map between domain memories and their PostgreSQL rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, memory: Memory) -> None:
        self._session.add(
            MemoryRow(
                memory_id=memory.memory_id,
                message_id=memory.message_id,
                content=memory.content,
                created_at=memory.created_at,
                valid_from=memory.valid_from,
                valid_to=memory.valid_to,
                session_scope=memory.session_scope,
            )
        )

    def get(self, *, memory_id: UUID, user_id: UUID) -> Memory | None:
        statement = (
            select(MemoryRow)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                MemoryRow.memory_id == memory_id,
                SessionRow.user_id == user_id,
            )
        )
        row = self._session.scalar(statement)
        if row is None:
            return None
        return Memory(
            memory_id=row.memory_id,
            message_id=row.message_id,
            content=row.content,
            created_at=row.created_at,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            session_scope=row.session_scope,
        )
