from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine

from fluxmem.adapters.postgres.database import (
    create_database_engine,
    create_session_factory,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    RetrievalForAdding,
    RetrievalForAnswering,
)
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.ports.lifecycle import LifecycleEvaluator
from fluxmem.application.ports.embeddings import EmbeddingProvider
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.application.write.reindex_memories import ReindexPendingMemories
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
    reinforce_memory: ReinforceMemory
    reindex_pending_memories: ReindexPendingMemories | None

    def close(self) -> None:
        self.engine.dispose()


def bootstrap(
    *,
    database_url: str,
    lifecycle_evaluator: LifecycleEvaluator | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    retrieval_settings: HybridRetrievalSettings | None = None,
    **engine_options: object,
) -> FluxMemServices:
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
            unit_of_work_factory=unit_of_work_factory,
            embedding_provider=embedding_provider,
            settings=retrieval_settings,
        ),
        retrieval_for_adding=RetrievalForAdding(
            unit_of_work_factory=unit_of_work_factory,
            embedding_provider=embedding_provider,
            settings=retrieval_settings,
        ),
        store_message=StoreMessage(unit_of_work_factory=unit_of_work_factory),
        store_memory=StoreMemory(
            unit_of_work_factory=unit_of_work_factory,
            lifecycle_evaluator=lifecycle_evaluator,
            embedding_provider=embedding_provider,
        ),
        reinforce_memory=ReinforceMemory(
            unit_of_work_factory=unit_of_work_factory,
        ),
        reindex_pending_memories=(
            ReindexPendingMemories(
                unit_of_work_factory=unit_of_work_factory,
                embedding_provider=embedding_provider,
            )
            if embedding_provider is not None
            else None
        ),
    )
