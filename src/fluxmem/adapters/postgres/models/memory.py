from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from fluxmem.adapters.postgres.models.base import Base
from fluxmem.domain.retrieval import EMBEDDING_DIMENSIONS


class MemoryRow(Base):
    """Database representation of an append-only domain memory."""

    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint(
            "valid_from IS NULL OR valid_to IS NULL OR valid_to >= valid_from",
            name="memories_valid_interval",
        ),
    )

    memory_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.message_id"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    session_applicability: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sessions.session_id"),
        nullable=True,
    )

class MemoryIndexRow(Base):
    """Database representation of a memory's retrieval projection."""

    __tablename__ = "memory_indexes"
    __table_args__ = (
        CheckConstraint(
            "index_status IN ('ready', 'pending')",
            name="memory_indexes_status",
        ),
        CheckConstraint(
            "(index_status = 'ready' AND embedding IS NOT NULL "
            "AND embedding_model IS NOT NULL) OR "
            "(index_status = 'pending' AND embedding IS NULL "
            "AND embedding_model IS NULL)",
            name="memory_indexes_state",
        ),
    )

    memory_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="CASCADE"),
        primary_key=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS).with_variant(JSON(), "sqlite"),
        nullable=True,
    )

    search_text: Mapped[str] = mapped_column(
        TSVECTOR().with_variant(Text(), "sqlite"),
        nullable=False,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    index_status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pending",
        server_default="pending",
    )

    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


Index("memories_by_message", MemoryRow.message_id, MemoryRow.memory_id)
Index(
    "memories_by_session_applicability",
    MemoryRow.session_applicability,
    MemoryRow.memory_id,
)

Index(
    "memory_indexes_search_text_gin",
    MemoryIndexRow.search_text,
    postgresql_using="gin",
)

Index(
    "memory_indexes_embedding_hnsw",
    MemoryIndexRow.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
