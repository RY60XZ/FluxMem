from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from fluxmem import (
    FluxMem,
    MemoryPack,
    MemoryRetrievalResult,
    Message,
    MessageIngestionResult,
    MessagePack,
    ModelTokenUsage,
)
from fluxmem_infrastructure.answering.generator import AnswerGenerator


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: Message
    retrieval: MemoryRetrievalResult
    context_memory_ids: tuple[UUID, ...]
    model: str
    response_id: str | None
    usage: ModelTokenUsage | None

    def __post_init__(self) -> None:
        if self.answer.session_id != self.retrieval.query.session_id:
            raise ValueError("query and answer must share one session")
        returned_ids = {
            item.memory.memory_id for item in self.retrieval.context.memories
        }
        if len(set(self.context_memory_ids)) != len(self.context_memory_ids):
            raise ValueError("answer context memory IDs must be unique")
        if not set(self.context_memory_ids).issubset(returned_ids):
            raise ValueError("answer context must come from retrieved memories")

    @property
    def query(self) -> Message:
        return self.retrieval.query


@dataclass(frozen=True, slots=True)
class AnswerTurnResult:
    answer_result: AnswerResult
    ingestion: MessageIngestionResult


class AnsweringAgent:
    """Example external agent composed solely through FluxMem's public API."""

    def __init__(
        self,
        *,
        memory: FluxMem,
        generator: AnswerGenerator,
        include_history: bool = True,
        include_memories: bool = True,
    ) -> None:
        self._memory = memory
        self._generator = generator
        self._include_history = include_history
        self._include_memories = include_memories

    def answer(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        created_at: datetime | None = None,
        retrieval_limit: int | None = None,
    ) -> AnswerResult:
        retrieval = (
            self._memory.retrieve(
                user_id=user_id,
                session_id=session_id,
                query=content,
                created_at=created_at,
                limit=retrieval_limit,
            )
            if self._include_memories
            else _empty_retrieval(
                user_id=user_id,
                session_id=session_id,
                content=content,
                created_at=created_at,
            )
        )
        history = (
            self._memory.get_session_history(
                user_id=user_id,
                session_id=session_id,
            )
            if self._include_history
            else MessagePack(
                user_id=user_id,
                session_id=session_id,
                messages=(),
            )
        )
        generated = self._generator.generate(
            query=retrieval.query,
            history=history,
            memory_pack=retrieval.context,
        )
        answer = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="assistant",
            agent_id=generated.model,
            content=generated.content,
            created_at=created_at or datetime.now().astimezone(),
        )
        return AnswerResult(
            answer=answer,
            retrieval=retrieval,
            context_memory_ids=generated.context_memory_ids,
            model=generated.model,
            response_id=generated.response_id,
            usage=generated.usage,
        )

    def run_turn(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        content: str,
        created_at: datetime | None = None,
    ) -> AnswerTurnResult:
        result = self.answer(
            user_id=user_id,
            session_id=session_id,
            content=content,
            created_at=created_at,
        )
        if self._include_memories:
            self._memory.record_usage(
                user_id=user_id,
                session_id=session_id,
                query_id=result.retrieval.query_id,
                used_memory_ids=result.context_memory_ids,
            )
        ingestion = self._memory.ingest_messages(
            user_id=user_id,
            messages=(result.query,),
        )
        self._memory.store_messages(
            user_id=user_id,
            messages=(result.answer,),
        )
        return AnswerTurnResult(answer_result=result, ingestion=ingestion)


def _empty_retrieval(
    *,
    user_id: UUID,
    session_id: UUID,
    content: str,
    created_at: datetime | None,
) -> MemoryRetrievalResult:
    query = Message(
        message_id=uuid4(),
        session_id=session_id,
        role="user",
        agent_id=None,
        content=content,
        created_at=created_at or datetime.now().astimezone(),
    )
    pack = MemoryPack(
        query_id=uuid4(),
        user_id=user_id,
        session_id=session_id,
        memories=(),
    )
    return MemoryRetrievalResult(
        query=query,
        context=pack,
    )
