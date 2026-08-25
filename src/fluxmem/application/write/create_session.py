from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.message import Session


class CreateSession:
    """Create a local user when needed and provision one owned session."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._id_factory = id_factory

    def execute(
        self,
        *,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> Session:
        conversation_session = Session(
            session_id=session_id or self._id_factory(),
            user_id=user_id,
        )
        with self._unit_of_work_factory() as unit_of_work:
            unit_of_work.sessions.ensure_user(user_id=user_id)
            unit_of_work.flush()
            if unit_of_work.sessions.is_owned_by(
                session_id=conversation_session.session_id,
                user_id=user_id,
            ):
                return conversation_session
            unit_of_work.sessions.add(session=conversation_session)
            unit_of_work.commit()
        return conversation_session
