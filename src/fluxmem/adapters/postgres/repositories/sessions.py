from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.session import SessionRow, UserRow
from fluxmem.domain.message import Session as ConversationSession


class SqlAlchemySessionRepository:
    """Resolve session ownership through one SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure_user(self, *, user_id: UUID) -> None:
        if self._session.get(UserRow, user_id) is None:
            self._session.add(UserRow(user_id=user_id))

    def add(self, *, session: ConversationSession) -> None:
        self._session.add(
            SessionRow(
                session_id=session.session_id,
                user_id=session.user_id,
            )
        )

    def is_owned_by(self, *, session_id: UUID, user_id: UUID) -> bool:
        statement = (
            select(SessionRow.session_id)
            .where(
                SessionRow.session_id == session_id,
                SessionRow.user_id == user_id,
            )
            .limit(1)
        )
        return self._session.scalar(statement) is not None
