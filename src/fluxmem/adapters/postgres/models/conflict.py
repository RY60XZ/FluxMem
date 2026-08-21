from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from fluxmem.adapters.postgres.models.base import Base


class MemoryConflictRow(Base):
    """One undirected potential-conflict edge stored in canonical order."""

    __tablename__ = "memory_conflicts"
    __table_args__ = (
        CheckConstraint(
            "memory_a_id < memory_b_id",
            name="memory_conflicts_canonical_order",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="memory_conflicts_confidence",
        ),
    )

    memory_a_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    memory_b_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


Index(
    "memory_conflicts_by_b",
    MemoryConflictRow.memory_b_id,
    MemoryConflictRow.memory_a_id,
)
