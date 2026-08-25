from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fluxmem.application.errors import SessionNotFoundError
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.message import Message


class StoreMessage:
    """Persist one message after validating ownership of its session."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    def execute(self, *, user_id: UUID, message: Message) -> UUID:
        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=message.session_id,
                user_id=user_id,
            ):
                raise SessionNotFoundError(
                    "message session does not belong to the requested user"
                )

            existing = unit_of_work.messages.get(
                message_id=message.message_id,
                user_id=user_id,
            )
            if existing is not None:
                if existing != message:
                    raise ValueError(
                        "message ID already exists with different content"
                    )
                return message.message_id

            unit_of_work.messages.add(message=message)
            unit_of_work.commit()

        return message.message_id
