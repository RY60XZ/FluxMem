from __future__ import annotations

from collections.abc import Callable, Iterable
from uuid import UUID, uuid4

from fluxmem.application.errors import (
    InvalidRetrievalContextError,
    SessionNotFoundError,
)
from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.message import Message


def _validate_context_message(*, message: Message, history: MessagePack) -> None:
    if message.session_id != history.session_id:
        raise InvalidRetrievalContextError(
            "retrieval message and session history belong to different sessions"
        )


def _extend_message_pack(
    *,
    history: MessagePack,
    messages: Iterable[Message],
) -> MessagePack:
    """Add messages without losing metadata or duplicating message identities."""

    combined_messages: list[Message] = []
    seen_message_ids: set[UUID] = set()
    for message in (*history.messages, *messages):
        if message.message_id in seen_message_ids:
            continue
        seen_message_ids.add(message.message_id)
        combined_messages.append(message)

    return MessagePack(
        user_id=history.user_id,
        session_id=history.session_id,
        messages=tuple(combined_messages),
    )


class RetrievalForAnswering:
    """Retrieve memories relevant before the answering model is invoked."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        clock: Clock | None = None,
        query_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._clock = clock or SystemClock()
        self._query_id_factory = query_id_factory

    def execute(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        limit: int = 10,
    ) -> MemoryPack:
        _validate_context_message(message=message, history=session_history)
        if limit < 1:
            raise ValueError("retrieval limit must be positive")

        retrieval_messages = _extend_message_pack(
            history=session_history,
            messages=(message,),
        )
        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=session_history.session_id,
                user_id=session_history.user_id,
            ):
                raise SessionNotFoundError(
                    "retrieval session does not belong to the requested user"
                )

            memories = unit_of_work.memories.search(
                message_pack=retrieval_messages,
                as_of=self._clock.now(),
                limit=limit,
            )

        return MemoryPack(
            query_id=self._query_id_factory(),
            user_id=session_history.user_id,
            session_id=session_history.session_id,
            memories=memories,
        )


class RetrievalForAdding:
    """Retrieve write-time context after the answering model has responded."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        clock: Clock | None = None,
        query_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._clock = clock or SystemClock()
        self._query_id_factory = query_id_factory

    def execute(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        model_response: Message,
        limit: int = 10,
    ) -> MemoryPack:
        _validate_context_message(message=message, history=session_history)
        _validate_context_message(message=model_response, history=session_history)
        if limit < 1:
            raise ValueError("retrieval limit must be positive")

        retrieval_messages = _extend_message_pack(
            history=session_history,
            messages=(message, model_response),
        )
        with self._unit_of_work_factory() as unit_of_work:
            if not unit_of_work.sessions.is_owned_by(
                session_id=session_history.session_id,
                user_id=session_history.user_id,
            ):
                raise SessionNotFoundError(
                    "retrieval session does not belong to the requested user"
                )

            memories = unit_of_work.memories.search(
                message_pack=retrieval_messages,
                as_of=self._clock.now(),
                limit=limit,
            )

        return MemoryPack(
            query_id=self._query_id_factory(),
            user_id=session_history.user_id,
            session_id=session_history.session_id,
            memories=memories,
        )
