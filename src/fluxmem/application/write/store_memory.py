from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fluxmem.application.errors import (
    MessageNotFoundError,
)
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.memory import Memory


class StoreMemory:
    """Persist one domain memory after validating its ownership boundaries."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    def execute(self, *, user_id: UUID, memory: Memory) -> UUID:
        with self._unit_of_work_factory() as unit_of_work:
            message = unit_of_work.messages.get(
                message_id=memory.message_id,
                user_id=user_id,
            )
            if message is None:
                raise MessageNotFoundError(
                    "origin message does not belong to the requested user"
                )

            unit_of_work.memories.add(memory=memory)
            unit_of_work.commit()

        return memory.memory_id
