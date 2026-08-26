from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.read.retrieval import HybridMemoryRetriever
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.domain.info_pack import MemoryRetrievalResult
from fluxmem.domain.message import Message


class RetrieveMemories:
    """Return memory context for an ephemeral external-service query."""

    def __init__(
        self,
        *,
        get_session_history: GetSessionHistory,
        retriever: HybridMemoryRetriever,
        clock: Clock | None = None,
        id_factory: Callable[[], UUID] = uuid4,
        history_limit: int = 128,
        result_limit: int = 10,
    ) -> None:
        if history_limit < 1 or result_limit < 1:
            raise ValueError("retrieval limits must be positive")
        self._get_session_history = get_session_history
        self._retriever = retriever
        self._clock = clock or SystemClock()
        self._id_factory = id_factory
        self._history_limit = history_limit
        self._result_limit = result_limit

    def execute(
        self,
        *,
        user_id: UUID,
        query: Message,
        limit: int | None = None,
    ) -> MemoryRetrievalResult:
        if query.role != "user":
            raise ValueError("memory query must have role 'user'")
        if not query.content.strip():
            raise ValueError("memory query cannot be blank")
        result_limit = self._result_limit if limit is None else limit
        if result_limit < 1:
            raise ValueError("retrieval limit must be positive")
        history = self._get_session_history.execute(
            user_id=user_id,
            session_id=query.session_id,
            limit=self._history_limit,
        )
        return MemoryRetrievalResult(
            query=query,
            context=self._retriever.execute(
                message=query,
                session_history=history,
                limit=result_limit,
            ),
        )

    def execute_text(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        created_at: datetime | None = None,
        limit: int | None = None,
    ) -> MemoryRetrievalResult:
        return self.execute(
            user_id=user_id,
            query=Message(
                message_id=self._id_factory(),
                session_id=session_id,
                role="user",
                agent_id=None,
                content=content,
                created_at=created_at or self._clock.now(),
            ),
            limit=limit,
        )
