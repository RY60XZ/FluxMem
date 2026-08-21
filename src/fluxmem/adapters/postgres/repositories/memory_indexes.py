from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.memory import MemoryIndexRow, MemoryRow
from fluxmem.domain.memory import Memory
from fluxmem.domain.retrieval import Embedding, IndexStatus, MemoryIndex


def _to_domain(row: MemoryRow) -> Memory:
    return Memory(
        memory_id=row.memory_id,
        message_id=row.message_id,
        content=row.content,
        created_at=row.created_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        session_applicability=row.session_applicability,
    )


class SqlAlchemyMemoryIndexRepository:
    """Persist and repair PostgreSQL lexical/vector projections."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, *, index: MemoryIndex) -> None:
        embedding = index.embedding
        self._session.add(
            MemoryIndexRow(
                memory_id=index.memory_id,
                embedding=list(embedding.values) if embedding else None,
                search_text=func.to_tsvector("english", index.text),
                embedding_model=embedding.model if embedding else None,
                index_status=index.status.value,
                indexed_at=index.indexed_at,
            )
        )

    def list_pending(self, *, limit: int) -> tuple[Memory, ...]:
        if limit < 1:
            raise ValueError("pending-index limit must be positive")
        statement = (
            select(MemoryRow)
            .join(
                MemoryIndexRow,
                MemoryIndexRow.memory_id == MemoryRow.memory_id,
            )
            .where(MemoryIndexRow.index_status == IndexStatus.PENDING.value)
            .order_by(MemoryIndexRow.indexed_at, MemoryRow.memory_id)
            .limit(limit)
        )
        return tuple(_to_domain(row) for row in self._session.scalars(statement))

    def mark_ready(
        self,
        *,
        memory_id: UUID,
        embedding: Embedding,
        indexed_at: datetime,
    ) -> bool:
        statement = (
            update(MemoryIndexRow)
            .where(
                MemoryIndexRow.memory_id == memory_id,
                MemoryIndexRow.index_status == IndexStatus.PENDING.value,
            )
            .values(
                embedding=list(embedding.values),
                embedding_model=embedding.model,
                index_status=IndexStatus.READY.value,
                indexed_at=indexed_at,
            )
        )
        result = self._session.execute(statement)
        return result.rowcount == 1
