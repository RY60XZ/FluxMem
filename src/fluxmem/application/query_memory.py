from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem.application.diagnostics import TurnDiagnosticsCollector
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.llm import AnswerGenerator
from fluxmem.application.read.retrieval import RetrievalForAnswering
from fluxmem.application.read.session_history import GetSessionHistory
from fluxmem.domain.llm import MemoryQueryResult
from fluxmem.domain.message import Message


class QueryMemory:
    """Answer an ephemeral query without learning from or reinforcing it."""

    def __init__(
        self,
        *,
        get_session_history: GetSessionHistory,
        retrieval_for_answering: RetrievalForAnswering,
        answer_generator: AnswerGenerator,
        clock: Clock | None = None,
        id_factory: Callable[[], UUID] = uuid4,
        history_limit: int = 128,
        answering_limit: int = 10,
    ) -> None:
        if history_limit < 1 or answering_limit < 1:
            raise ValueError("query limits must be positive")
        self._get_session_history = get_session_history
        self._retrieval_for_answering = retrieval_for_answering
        self._answer_generator = answer_generator
        self._clock = clock or SystemClock()
        self._id_factory = id_factory
        self._history_limit = history_limit
        self._answering_limit = answering_limit

    def execute(
        self,
        *,
        user_id: UUID,
        message: Message,
        agent_id: str | None = None,
        diagnostics: bool = False,
    ) -> MemoryQueryResult:
        if message.role != "user":
            raise ValueError("memory query input must have role 'user'")
        if not message.content.strip():
            raise ValueError("memory query content cannot be blank")

        history = self._get_session_history.execute(
            user_id=user_id,
            session_id=message.session_id,
            limit=self._history_limit,
        )
        retrieval = self._retrieval_for_answering.execute(
            message=message,
            session_history=history,
            limit=self._answering_limit,
        )
        diagnostics_collector = TurnDiagnosticsCollector() if diagnostics else None
        if diagnostics_collector is not None:
            diagnostics_collector.record_retrieval(retrieval)
        usage_collector = ModelUsageCollector()
        generated = self._answer_generator.generate(
            message=message,
            session_history=history,
            memory_pack=retrieval.expanded,
            usage_recorder=usage_collector,
            diagnostics_recorder=diagnostics_collector,
        )
        if diagnostics_collector is not None:
            diagnostics_collector.record_answer_context(
                generated.context_memory_ids
            )
            diagnostics_collector.complete(
                feedback_applied=False,
                memory_outcomes=(),
                post_answer_errors=(),
            )
        answer = Message(
            message_id=self._id_factory(),
            session_id=message.session_id,
            role="assistant",
            agent_id=agent_id,
            content=generated.content,
            created_at=self._clock.now(),
        )
        return MemoryQueryResult(
            query=message,
            answer=answer,
            answering_memory_pack=retrieval.expanded,
            context_memory_ids=generated.context_memory_ids,
            llm_usage=usage_collector.snapshot(),
            diagnostics=(
                diagnostics_collector.snapshot()
                if diagnostics_collector is not None
                else None
            ),
        )

    def execute_text(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        agent_id: str | None = None,
        diagnostics: bool = False,
        created_at: datetime | None = None,
    ) -> MemoryQueryResult:
        return self.execute(
            user_id=user_id,
            message=Message(
                message_id=self._id_factory(),
                session_id=session_id,
                role="user",
                agent_id=None,
                content=content,
                created_at=created_at or self._clock.now(),
            ),
            agent_id=agent_id,
            diagnostics=diagnostics,
        )
