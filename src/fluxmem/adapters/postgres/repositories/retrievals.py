from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.retrieval import (
    RetrievalCandidateRow,
    RetrievalQueryRow,
)
from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.info_pack import RetrievedMemory
from fluxmem.domain.lifecycle import Status
from fluxmem.domain.retrieval import QueryType


class SqlAlchemyRetrievalRepository:
    """Persist retrieval events and resolve their scoped candidate sets."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        query_id: UUID,
        session_id: UUID,
        query_type: QueryType,
        candidates: tuple[RetrievedMemory, ...],
        created_at: datetime,
    ) -> None:
        self._session.add(
            RetrievalQueryRow(
                query_id=query_id,
                session_id=session_id,
                query_type=query_type.value,
                created_at=created_at,
            )
        )
        # The row mappings intentionally have no ORM relationship. Flush the
        # parent explicitly so PostgreSQL always sees it before candidate FKs.
        self._session.flush()
        self._session.add_all(
            RetrievalCandidateRow(
                query_id=query_id,
                memory_id=candidate.memory.memory_id,
                rank=candidate.rank,
                score=candidate.score,
            )
            for candidate in candidates
        )

    def candidate_ids_for_query(
        self,
        *,
        query_id: UUID,
        user_id: UUID,
        session_id: UUID,
        query_type: QueryType | None = None,
    ) -> tuple[UUID, ...] | None:
        ownership_statement = (
            select(RetrievalQueryRow.query_id)
            .join(
                SessionRow,
                SessionRow.session_id == RetrievalQueryRow.session_id,
            )
            .where(
                RetrievalQueryRow.query_id == query_id,
                RetrievalQueryRow.session_id == session_id,
                SessionRow.user_id == user_id,
            )
        )
        if query_type is not None:
            ownership_statement = ownership_statement.where(
                RetrievalQueryRow.query_type == query_type.value
            )
        owned_query = self._session.scalar(ownership_statement)
        if owned_query is None:
            return None

        statement = (
            select(RetrievalCandidateRow.memory_id)
            .where(RetrievalCandidateRow.query_id == query_id)
            .order_by(
                RetrievalCandidateRow.rank,
                RetrievalCandidateRow.memory_id,
            )
        )
        if query_type in (QueryType.ANSWERING, QueryType.ADDING):
            statement = statement.join(
                MemoryLifecycleRow,
                MemoryLifecycleRow.memory_id == RetrievalCandidateRow.memory_id,
            ).where(MemoryLifecycleRow.status == Status.ACTIVE.value)
        return tuple(self._session.scalars(statement))
