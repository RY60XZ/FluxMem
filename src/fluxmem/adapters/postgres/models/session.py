from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from fluxmem.adapters.postgres.models.base import Base


class UserRow(Base):
    """Database representation of a domain user identity."""

    __tablename__ = "users"

    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )


class SessionRow(Base):
    """A conversation session with exactly one owning user."""

    __tablename__ = "sessions"

    session_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.user_id"),
        nullable=False,
    )


Index("sessions_by_user", SessionRow.user_id, SessionRow.session_id)
