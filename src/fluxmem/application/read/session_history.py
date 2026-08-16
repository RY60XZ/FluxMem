from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fluxmem.application.errors import SessionNotFoundError
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.info_pack import MessagePack


class GetSessionHistory:
    """Load the chronological message history of one user-owned session."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    def execute(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int | None = None,
    ) -> MessagePack:
        if limit is not None and limit < 1:
            raise ValueError("session-history limit must be positive")

        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=session_id,
                user_id=user_id,
            ):
                raise SessionNotFoundError(
                    "session history does not belong to the requested user"
                )

            messages = unit_of_work.messages.list_for_session(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
            )

        return MessagePack(
            user_id=user_id,
            session_id=session_id,
            messages=messages,
        )
