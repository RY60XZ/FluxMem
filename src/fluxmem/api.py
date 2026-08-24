from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import Engine

from fluxmem.application.memory_learning import LearnFromMessages
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.application.retrieve import RetrieveMemories
from fluxmem.application.write.create_session import CreateSession
from fluxmem.application.write.reindex_memories import (
    ReindexPendingMemories,
    ReindexSummary,
)
from fluxmem.application.write.reinforcement import ReinforceMemory
from fluxmem.application.write.store_message import StoreMessage
from fluxmem.domain.info_pack import (
    FeedbackPack,
    MemoryRetrievalResult,
    MemoryUsage,
    MessagePack,
    UsageType,
)
from fluxmem.domain.lifecycle import MemoryLifecycle
from fluxmem.domain.llm import MessageIngestionResult
from fluxmem.domain.message import Message, Session


class FluxMem:
    """Stable, provider-neutral facade for the independent memory layer."""

    def __init__(
        self,
        *,
        engine: Engine,
        create_session: CreateSession,
        get_session_history: GetSessionHistory,
        store_message: StoreMessage,
        retrieve_memories: RetrieveMemories,
        reinforce_memory: ReinforceMemory,
        learn_from_messages: LearnFromMessages | None,
        reindex_pending: ReindexPendingMemories | None,
    ) -> None:
        self._engine = engine
        self._create_session = create_session
        self._get_session_history = get_session_history
        self._store_message = store_message
        self._retrieve_memories = retrieve_memories
        self._reinforce_memory = reinforce_memory
        self._learn_from_messages = learn_from_messages
        self._reindex_pending = reindex_pending

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
        return self._create_session.execute(
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
        if self._learn_from_messages is None:
            raise RuntimeError("memory learning is not configured")
        return self._learn_from_messages.execute(
            user_id=user_id,
            messages=messages,
            diagnostics=diagnostics,
        )

    def store_messages(
        self,
        *,
        user_id: UUID,
        messages: Sequence[Message],
    ) -> tuple[UUID, ...]:
        """Persist messages without extracting or modifying memories."""

        return tuple(
            self._store_message.execute(user_id=user_id, message=message)
            for message in messages
        )

    def retrieve(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        query: str,
        created_at: datetime | None = None,
        limit: int | None = None,
    ) -> MemoryRetrievalResult:
        return self._retrieve_memories.execute_text(
            user_id=user_id,
            session_id=session_id,
            content=query,
            created_at=created_at,
            limit=limit,
        )

    def get_session_history(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int | None = None,
    ) -> MessagePack:
        return self._get_session_history.execute(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

    def record_usage(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        query_id: UUID,
        used_memory_ids: Sequence[UUID],
    ) -> tuple[MemoryLifecycle, ...]:
        return self._reinforce_memory.execute(
            feedback_pack=FeedbackPack(
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                used_memories=tuple(
                    MemoryUsage(
                        memory_id=memory_id,
                        usage_type=UsageType.CONTEXT_INCLUDED,
                    )
                    for memory_id in used_memory_ids
                ),
            )
        )

    def reindex_pending_memories(self, *, limit: int = 100) -> ReindexSummary:
        if self._reindex_pending is None:
            raise RuntimeError("memory reindexing is not configured")
        return self._reindex_pending.execute(limit=limit)

    def close(self) -> None:
        self._engine.dispose()
