from __future__ import annotations

from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker

from fluxmem.adapters.postgres.repositories.memories import (
    SqlAlchemyMemoryRepository,
)
from fluxmem.adapters.postgres.repositories.memory_indexes import (
    SqlAlchemyMemoryIndexRepository,
)
from fluxmem.adapters.postgres.repositories.lifecycles import (
    SqlAlchemyLifecycleRepository,
)
from fluxmem.adapters.postgres.repositories.messages import (
    SqlAlchemyMessageRepository,
)
from fluxmem.adapters.postgres.repositories.sessions import (
    SqlAlchemySessionRepository,
)
from fluxmem.adapters.postgres.repositories.retrievals import (
    SqlAlchemyRetrievalRepository,
)


class SqlAlchemyUnitOfWork:
    """Share one transaction across all PostgreSQL repositories."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self._session = self._session_factory()
        self.lifecycles = SqlAlchemyLifecycleRepository(self._session)
        self.messages = SqlAlchemyMessageRepository(self._session)
        self.sessions = SqlAlchemySessionRepository(self._session)
        self.memories = SqlAlchemyMemoryRepository(self._session)
        self.memory_indexes = SqlAlchemyMemoryIndexRepository(self._session)
        self.retrievals = SqlAlchemyRetrievalRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.rollback()
        self._session.close()

    def commit(self) -> None:
        self._session.commit()

    def flush(self) -> None:
        self._session.flush()

    def rollback(self) -> None:
        self._session.rollback()
