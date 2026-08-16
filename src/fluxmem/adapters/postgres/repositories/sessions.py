from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.session import SessionRow


class SqlAlchemySessionRepository:
    """Resolve session ownership through one SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

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
