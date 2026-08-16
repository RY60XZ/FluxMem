from __future__ import annotations

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.session import SessionRow
from fluxmem.domain.message import Message


def _to_domain(row: MessageRow) -> Message:
    return Message(
        message_id=row.message_id,
        session_id=row.session_id,
        role=row.role,
        agent_id=row.agent_id,
        content=row.content,
        created_at=row.created_at,
    )


def session_history(
    *,
    user_id: UUID,
    session_id: UUID,
    limit: int | None = None,
) -> Select[tuple[MessageRow]]:
    """Select all or the latest N owned messages in chronological order."""

    owned_messages = (
        select(MessageRow)
        .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
        .where(SessionRow.user_id == user_id, MessageRow.session_id == session_id)
    )
    if limit is None:
        return owned_messages.order_by(MessageRow.created_at, MessageRow.message_id)

    recent_message_ids = (
        select(MessageRow.message_id)
        .join(SessionRow, SessionRow.session_id == MessageRow.session_id)
        .where(SessionRow.user_id == user_id, MessageRow.session_id == session_id)
        .order_by(MessageRow.created_at.desc(), MessageRow.message_id.desc())
        .limit(limit)
        .subquery()
    )
    return (
        select(MessageRow)
        .join(
            recent_message_ids,
            recent_message_ids.c.message_id == MessageRow.message_id,
        )
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
        return _to_domain(row)

    def add(self, *, message: Message) -> None:
        self._session.add(
            MessageRow(
                message_id=message.message_id,
                session_id=message.session_id,
                role=message.role,
                agent_id=message.agent_id,
                content=message.content,
                created_at=message.created_at,
            )
        )

    def list_for_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int | None = None,
    ) -> tuple[Message, ...]:
        rows = self._session.scalars(
            session_history(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
            )
        )
        return tuple(_to_domain(row) for row in rows)
