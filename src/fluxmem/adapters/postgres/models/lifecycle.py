from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from fluxmem.adapters.postgres.models.base import Base


class MemoryLifecycleRow(Base):
    """Mutable lifecycle state for exactly one canonical memory."""

    __tablename__ = "memory_lifecycle"
    __table_args__ = (
        CheckConstraint(
            "tier IN ('working', 'short_term', 'long_term')",
            name="memory_lifecycle_tier",
        ),
        CheckConstraint(
            "status IN ('active', 'deleted')",
            name="memory_lifecycle_status",
        ),
        CheckConstraint(
            "importance >= 0.1 AND importance <= 1",
            name="memory_lifecycle_importance",
        ),
        CheckConstraint(
            "decay_class IN ('fast', 'standard', 'slow')",
            name="memory_lifecycle_decay_class",
        ),
        CheckConstraint(
            "retention_snapshot >= 0.1 AND retention_snapshot <= 1",
            name="memory_lifecycle_retention",
        ),
        CheckConstraint(
            "reinforcement_count >= 0 AND use_count >= 0",
            name="memory_lifecycle_counters",
        ),
        CheckConstraint(
            "decision_source IN "
            "('llm_primary', 'llm_retry', 'rules_fallback', 'manual')",
            name="memory_lifecycle_decision_source",
        ),
    )

    memory_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="CASCADE"),
        primary_key=True,
    )
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    decay_class: Mapped[str] = mapped_column(Text, nullable=False)
    retention_snapshot: Mapped[float] = mapped_column(Float, nullable=False)
    retention_anchor: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    reinforcement_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    decision_source: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


Index(
    "memory_lifecycle_by_status",
    MemoryLifecycleRow.status,
    MemoryLifecycleRow.memory_id,
)
