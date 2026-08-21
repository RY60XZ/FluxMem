from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from fluxmem.adapters.postgres.models.base import Base


class RetrievalQueryRow(Base):
    """One user-scoped retrieval event, including empty result sets."""

    __tablename__ = "retrieval_queries"
    __table_args__ = (
        CheckConstraint(
            "query_type IN ('answering', 'adding')",
            name="retrieval_queries_type",
        ),
    )

    query_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    session_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sessions.session_id"),
        nullable=False,
    )
    query_type: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class RetrievalCandidateRow(Base):
    """A memory returned by one retrieval event."""

    __tablename__ = "retrieval_candidates"
    __table_args__ = (
        CheckConstraint("rank >= 1", name="retrieval_candidates_rank"),
        CheckConstraint("score >= 0", name="retrieval_candidates_score"),
        ForeignKeyConstraint(
            ["query_id"],
            ["retrieval_queries.query_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["memory_id"],
            ["memories.memory_id"],
            ondelete="CASCADE",
        ),
    )

    query_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    memory_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)


Index(
    "retrieval_queries_by_session",
    RetrievalQueryRow.session_id,
    RetrievalQueryRow.created_at,
    RetrievalQueryRow.query_id,
)
