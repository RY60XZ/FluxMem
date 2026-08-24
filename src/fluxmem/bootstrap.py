from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import Engine

from fluxmem.adapters.postgres.database import (
    create_database_engine,
    create_session_factory,
)
from fluxmem.adapters.postgres.unit_of_work import SqlAlchemyUnitOfWork
from fluxmem.application.llm import (
    LLMAnswerGenerator,
    LLMIntegrationSettings,
    LLMLifecycleEvaluator,
    LLMMemoryExtractor,
    LLMMemoryReconciler,
)
from fluxmem.application.memory_learning import LearnFromMessages, MemoryLearning
from fluxmem.application.ports.embeddings import EmbeddingProvider
from fluxmem.application.ports.lifecycle import LifecycleEvaluator
from fluxmem.application.ports.llm import ModelProvider
from fluxmem.application.process_turn import (
    ConversationTurnStream,
    ProcessConversationTurn,
)
from fluxmem.application.query_memory import QueryMemory
from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    RetrievalForAdding,
    RetrievalForAnswering,
)
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.write.create_session import CreateSession
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.application.write.reindex_memories import ReindexPendingMemories
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.application.write.store_message import StoreMessage
from fluxmem.domain.llm import (
    ConversationTurnResult,
    MemoryQueryResult,
    MessageIngestionResult,
)
from fluxmem.domain.message import Message, Session


@dataclass(frozen=True, slots=True)
class FluxMemServices:
    """Fully wired application services that share one database engine."""

    engine: Engine
    create_session: CreateSession
    get_session_history: GetSessionHistory
    retrieval_for_answering: RetrievalForAnswering
    retrieval_for_adding: RetrievalForAdding
    store_message: StoreMessage
    store_memory: StoreMemory
    reinforce_memory: ReinforceMemory
    reindex_pending_memories: ReindexPendingMemories | None
    process_turn: ProcessConversationTurn | None
    learn_from_messages: LearnFromMessages | None = None
    query_memory: QueryMemory | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def start_session(
        self,
        *,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> Session:
        """Create a conversation session for one application-owned user ID."""

        return self.create_session.execute(
            user_id=user_id,
            session_id=session_id,
        )

    def ingest_messages(
        self,
        *,
        user_id: UUID,
        messages: Sequence[Message],
        diagnostics: bool = False,
    ) -> MessageIngestionResult:
        """Learn from imported transcript messages without generating replies."""

        if self.learn_from_messages is None:
            raise RuntimeError("message learning is not configured")
        return self.learn_from_messages.execute(
            user_id=user_id,
            messages=messages,
            diagnostics=diagnostics,
        )

    def query(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_id: str | None = None,
        diagnostics: bool = False,
        created_at: datetime | None = None,
    ) -> MemoryQueryResult:
        """Answer without storing the exchange or changing memory lifecycle."""

        if self.query_memory is None:
            raise RuntimeError("memory querying is not configured")
        return self.query_memory.execute_text(
            user_id=user_id,
            session_id=session_id,
            content=content,
            agent_id=agent_id,
            diagnostics=diagnostics,
            created_at=created_at,
        )

    def stream_turn(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_id: str | None = None,
        diagnostics: bool = False,
    ) -> ConversationTurnStream:
        """Begin one plain-text turn through the configured model stack."""

        if self.process_turn is None:
            raise RuntimeError("conversation processing is not configured")
        return self.process_turn.execute_text(
            user_id=user_id,
            session_id=session_id,
            content=content,
            agent_id=agent_id,
            diagnostics=diagnostics,
        )

    def run_turn(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_id: str | None = None,
        diagnostics: bool = False,
        timeout: float | None = None,
    ) -> ConversationTurnResult:
        """Complete one text turn and wait for its post-answer memory work."""

        stream = self.stream_turn(
            user_id=user_id,
            session_id=session_id,
            content=content,
            agent_id=agent_id,
            diagnostics=diagnostics,
        )
        for _ in stream:
            pass
        return stream.wait_for_post_answer(timeout=timeout)

    def close(self) -> None:
        if self.process_turn is not None:
            self.process_turn.close()
        self.engine.dispose()


def bootstrap(
    *,
    database_url: str,
    lifecycle_evaluator: LifecycleEvaluator | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    retrieval_settings: HybridRetrievalSettings | None = None,
    structured_model_provider: ModelProvider | None = None,
    llm_settings: LLMIntegrationSettings | None = None,
    **engine_options: object,
) -> FluxMemServices:
    """Create FluxMem's PostgreSQL adapters and application use cases."""

    if (structured_model_provider is None) != (llm_settings is None):
        raise ValueError(
            "structured_model_provider and llm_settings must be configured together"
        )

    engine = create_database_engine(database_url, **engine_options)
    session_factory = create_session_factory(engine)

    def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    get_session_history = GetSessionHistory(
        unit_of_work_factory=unit_of_work_factory
    )
    create_session = CreateSession(unit_of_work_factory=unit_of_work_factory)
    retrieval_for_answering = RetrievalForAnswering(
        unit_of_work_factory=unit_of_work_factory,
        embedding_provider=embedding_provider,
        settings=retrieval_settings,
    )
    retrieval_for_adding = RetrievalForAdding(
        unit_of_work_factory=unit_of_work_factory,
        embedding_provider=embedding_provider,
        settings=retrieval_settings,
    )
    store_message = StoreMessage(unit_of_work_factory=unit_of_work_factory)
    reinforce_memory = ReinforceMemory(
        unit_of_work_factory=unit_of_work_factory,
    )

    answer_generator = None
    memory_extractor = None
    memory_reconciler = None
    if structured_model_provider is not None and llm_settings is not None:
        answer_generator = LLMAnswerGenerator(
            provider=structured_model_provider,
            settings=llm_settings.answer,
            context_settings=llm_settings.context,
        )
        memory_extractor = LLMMemoryExtractor(
            provider=structured_model_provider,
            settings=llm_settings.extraction,
            context_settings=llm_settings.context,
        )
        memory_reconciler = LLMMemoryReconciler(
            provider=structured_model_provider,
            settings=llm_settings.reconciliation,
            context_settings=llm_settings.context,
        )
        if lifecycle_evaluator is None and llm_settings.enable_llm_lifecycle:
            lifecycle_evaluator = LLMLifecycleEvaluator(
                provider=structured_model_provider,
                settings=llm_settings.lifecycle,
            )

    store_memory = StoreMemory(
        unit_of_work_factory=unit_of_work_factory,
        lifecycle_evaluator=lifecycle_evaluator,
        embedding_provider=embedding_provider,
    )
    process_turn = None
    learn_from_messages = None
    query_memory = None
    if (
        answer_generator is not None
        and memory_extractor is not None
        and memory_reconciler is not None
    ):
        process_turn = ProcessConversationTurn(
            get_session_history=get_session_history,
            retrieval_for_answering=retrieval_for_answering,
            store_message=store_message,
            store_memory=store_memory,
            reinforce_memory=reinforce_memory,
            answer_generator=answer_generator,
            memory_extractor=memory_extractor,
            memory_reconciler=memory_reconciler,
            history_limit=llm_settings.context.maximum_history_messages,
            enable_memory_extraction=llm_settings.enable_memory_extraction,
            enable_memory_writes=llm_settings.enable_memory_writes,
            enable_conflict_detection=llm_settings.enable_conflict_detection,
        )
        memory_learning = MemoryLearning(
            store_memory=store_memory,
            memory_extractor=memory_extractor,
            memory_reconciler=memory_reconciler,
            enable_memory_extraction=llm_settings.enable_memory_extraction,
            enable_memory_writes=llm_settings.enable_memory_writes,
            enable_conflict_detection=llm_settings.enable_conflict_detection,
        )
        learn_from_messages = LearnFromMessages(
            get_session_history=get_session_history,
            retrieval_for_answering=retrieval_for_answering,
            store_message=store_message,
            memory_learning=memory_learning,
            maximum_batch_messages=(
                llm_settings.context.maximum_write_history_messages
            ),
            history_limit=llm_settings.context.maximum_history_messages,
        )
        query_memory = QueryMemory(
            get_session_history=get_session_history,
            retrieval_for_answering=retrieval_for_answering,
            answer_generator=answer_generator,
            history_limit=llm_settings.context.maximum_history_messages,
        )

    return FluxMemServices(
        engine=engine,
        create_session=create_session,
        get_session_history=get_session_history,
        retrieval_for_answering=retrieval_for_answering,
        retrieval_for_adding=retrieval_for_adding,
        store_message=store_message,
        store_memory=store_memory,
        reinforce_memory=reinforce_memory,
        reindex_pending_memories=(
            ReindexPendingMemories(
                unit_of_work_factory=unit_of_work_factory,
                embedding_provider=embedding_provider,
            )
            if embedding_provider is not None
            else None
        ),
        process_turn=process_turn,
        learn_from_messages=learn_from_messages,
        query_memory=query_memory,
    )
