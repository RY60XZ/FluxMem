from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.memory import MemoryRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.info_pack import MessagePack, RetrievedMemory
from fluxmem.domain.memory import Memory


def _to_domain(row: MemoryRow) -> Memory:
    return Memory(
        memory_id=row.memory_id,
        message_id=row.message_id,
        content=row.content,
        created_at=row.created_at,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        session_scope=row.session_scope,
    )


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
        return _to_domain(row)

    def search(
        self,
        *,
        message_pack: MessagePack,
        as_of: datetime,
        limit: int,
    ) -> tuple[RetrievedMemory, ...]:
        """Run the initial lexical leg without discarding message metadata."""

        lexical_terms = tuple(
            dict.fromkeys(
                term
                for message in message_pack.messages
                for term in re.findall(r"\w+", message.content.casefold())
            )
        )[:64]
        if not lexical_terms:
            return ()

        search_document = func.to_tsvector("english", MemoryRow.content)
        search_query = func.websearch_to_tsquery(
            "english",
            " OR ".join(lexical_terms),
        )
        score = func.ts_rank_cd(search_document, search_query).label("score")
        statement = (
            select(MemoryRow, score)
            .join(MessageRow, MessageRow.message_id == MemoryRow.message_id)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                SessionRow.user_id == message_pack.user_id,
                or_(
                    MemoryRow.session_scope.is_(None),
                    MemoryRow.session_scope == message_pack.session_id,
                ),
                or_(MemoryRow.valid_from.is_(None), MemoryRow.valid_from <= as_of),
                or_(MemoryRow.valid_to.is_(None), MemoryRow.valid_to >= as_of),
                search_document.op("@@")(search_query),
            )
            .order_by(score.desc(), MemoryRow.created_at.desc(), MemoryRow.memory_id)
            .limit(limit)
        )
        rows = self._session.execute(statement).all()
        return tuple(
            RetrievedMemory(
                memory=_to_domain(row),
                rank=rank,
                score=float(row_score),
                retrieval_reasons=("lexical",),
            )
            for rank, (row, row_score) in enumerate(rows, start=1)
        )
