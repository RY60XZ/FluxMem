from __future__ import annotations

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.message import Message


def session_history(
    *,
    user_id: UUID,
    session_id: UUID,
) -> Select[tuple[MessageRow]]:
    """Select an owned session's history in chronological order."""

    return (
        select(MessageRow)
        .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
        .where(SessionRow.user_id == user_id, MessageRow.session_id == session_id)
        .order_by(MessageRow.created_at, MessageRow.message_id)
    )


class SqlAlchemyMessageRepository:
    """Read message evidence through one SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, *, message_id: UUID, user_id: UUID) -> Message | None:
        statement = (
            select(MessageRow)
            .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
            .where(
                MessageRow.message_id == message_id,
                SessionRow.user_id == user_id,
            )
        )
        row = self._session.scalar(statement)
        if row is None:
            return None
        return Message(
            message_id=row.message_id,
            session_id=row.session_id,
            role=row.role,
            agent_id=row.agent_id,
            content=row.content,
            created_at=row.created_at,
        )