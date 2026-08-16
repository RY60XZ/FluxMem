from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self

from fluxmem.application.ports.repositories import (
    LifecycleRepository,
    MemoryRepository,
    MessageRepository,
    SessionRepository,
)


class UnitOfWork(Protocol):
    lifecycles: LifecycleRepository
    messages: MessageRepository
    sessions: SessionRepository
    memories: MemoryRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def flush(self) -> None: ...

    def rollback(self) -> None: ...
