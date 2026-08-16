from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from fluxmem.adapters.postgres.models.base import Base


class MemoryUsageRow(Base):
    """One idempotent actual-use observation for a retrieved memory."""

    __tablename__ = "memory_usage"
    __table_args__ = (
        UniqueConstraint(
            "memory_id",
            "query_id",
            name="memory_usage_memory_query_key",
        ),
        CheckConstraint(
            "usage_type IN ('context_included', 'model_attributed')",
            name="memory_usage_type",
        ),
        CheckConstraint(
            "rank IS NULL OR rank >= 1",
            name="memory_usage_rank",
        ),
        CheckConstraint(
            "contribution IS NULL OR "
            "(contribution >= 0 AND contribution <= 1)",
            name="memory_usage_contribution",
        ),
    )

    usage_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    memory_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="CASCADE"),
        nullable=False,
    )
    query_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    usage_type: Mapped[str] = mapped_column(Text, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    contribution: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


Index(
    "memory_usage_by_memory_time",
    MemoryUsageRow.memory_id,
    MemoryUsageRow.used_at,
)
