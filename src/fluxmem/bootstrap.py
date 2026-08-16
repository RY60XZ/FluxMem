from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from fluxmem.adapters.postgres.database import (
    create_database_engine,
    create_session_factory,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.application.read.retrieval import (
    RetrievalForAdding,
    RetrievalForAnswering,
)
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.application.write.store_message import StoreMessage


@dataclass(frozen=True, slots=True)
class FluxMemServices:
    """Fully wired application services that share one database engine."""

    engine: Engine
    get_session_history: GetSessionHistory
    retrieval_for_answering: RetrievalForAnswering
    retrieval_for_adding: RetrievalForAdding
    store_message: StoreMessage
    store_memory: StoreMemory

    def close(self) -> None:
        self.engine.dispose()


def bootstrap(*, database_url: str, **engine_options: object) -> FluxMemServices:
    """Create FluxMem's PostgreSQL adapters and application use cases."""

    engine = create_database_engine(database_url, **engine_options)
    session_factory = create_session_factory(engine)

    def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return FluxMemServices(
        engine=engine,
        get_session_history=GetSessionHistory(
            unit_of_work_factory=unit_of_work_factory
        ),
        retrieval_for_answering=RetrievalForAnswering(
            unit_of_work_factory=unit_of_work_factory
        ),
        retrieval_for_adding=RetrievalForAdding(
            unit_of_work_factory=unit_of_work_factory
        ),
        store_message=StoreMessage(unit_of_work_factory=unit_of_work_factory),
        store_memory=StoreMemory(unit_of_work_factory=unit_of_work_factory),
    )
