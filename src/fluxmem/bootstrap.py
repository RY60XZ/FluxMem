from __future__ import annotations

from dataclasses import dataclass

from fluxmem.adapters.postgres.database import (
    create_database_engine,
    create_session_factory,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.api import FluxMem
from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.memory_learning import LearnFromMessages, MemoryLearning
from fluxmem.application.ports.embeddings import EmbeddingProvider
from fluxmem.application.ports.lifecycle import LifecycleEvaluator
from fluxmem.application.ports.llm import MemoryExtractor
from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    HybridMemoryRetriever,
)
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.retrieve import RetrieveMemories
from fluxmem.application.write.create_session import CreateSession
from fluxmem.application.write.reindex_memories import ReindexPendingMemories
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.application.write.store_message import StoreMessage


@dataclass(frozen=True, slots=True)
class MemoryLayerSettings:
    """Application limits and feature switches defined by the memory layer."""

    maximum_history_messages: int = 128
    maximum_ingestion_batch_messages: int = 6
    retrieval_limit: int = 50
    enable_memory_extraction: bool = True
    enable_memory_writes: bool = True

    def __post_init__(self) -> None:
        if (
            self.maximum_history_messages < 1
            or self.maximum_ingestion_batch_messages < 1
            or self.retrieval_limit < 1
        ):
            raise ValueError("memory-layer limits must be positive")


def bootstrap(
    *,
    database_url: str,
    memory_extractor: MemoryExtractor | None = None,
    lifecycle_evaluator: LifecycleEvaluator | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    retrieval_settings: HybridRetrievalSettings | None = None,
    settings: MemoryLayerSettings | None = None,
    **engine_options: object,
) -> FluxMem:
    """Build the provider-neutral FluxMem memory layer."""

    configured = settings or MemoryLayerSettings()
    engine = create_database_engine(database_url, **engine_options)
    session_factory = create_session_factory(engine)

    def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    get_session_history = GetSessionHistory(
        unit_of_work_factory=unit_of_work_factory
    )
    retriever = HybridMemoryRetriever(
        unit_of_work_factory=unit_of_work_factory,
        embedding_provider=embedding_provider,
        settings=retrieval_settings,
    )
    store_message = StoreMessage(unit_of_work_factory=unit_of_work_factory)
    store_memory = StoreMemory(
        unit_of_work_factory=unit_of_work_factory,
        embedding_provider=embedding_provider,
    )

    learn_from_messages = None
    if memory_extractor is not None:
        learn_from_messages = LearnFromMessages(
            get_session_history=get_session_history,
            retriever=retriever,
            store_message=store_message,
            memory_learning=MemoryLearning(
                store_memory=store_memory,
                memory_extractor=memory_extractor,
                lifecycle_assigner=LifecycleAssigner(
                    evaluator=lifecycle_evaluator,
                ),
                enable_memory_extraction=configured.enable_memory_extraction,
                enable_memory_writes=configured.enable_memory_writes,
            ),
            maximum_batch_messages=(
                configured.maximum_ingestion_batch_messages
            ),
            history_limit=configured.maximum_history_messages,
            retrieval_limit=configured.retrieval_limit,
        )

    return FluxMem(
        engine=engine,
        create_session=CreateSession(
            unit_of_work_factory=unit_of_work_factory
        ),
        get_session_history=get_session_history,
        store_message=store_message,
        retrieve_memories=RetrieveMemories(
            get_session_history=get_session_history,
            retriever=retriever,
            history_limit=configured.maximum_history_messages,
            result_limit=configured.retrieval_limit,
        ),
        reinforce_memory=ReinforceMemory(
            unit_of_work_factory=unit_of_work_factory
        ),
        learn_from_messages=learn_from_messages,
        reindex_pending=(
            ReindexPendingMemories(
                unit_of_work_factory=unit_of_work_factory,
                embedding_provider=embedding_provider,
            )
            if embedding_provider is not None
            else None
        ),
    )
