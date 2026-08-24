from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fluxmem.application.memory_learning import (
    LearnFromMessages,
    MemoryLearningResult,
)
from fluxmem.domain.info_pack import MemoryPack, MessagePack, TurnMemoryPacks
from fluxmem.domain.message import Message


class _StoreMessage:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def execute(self, *, user_id: UUID, message: Message) -> UUID:
        del user_id
        self.messages.append(message)
        return message.message_id


class _History:
    def __init__(self, stored: _StoreMessage) -> None:
        self._stored = stored

    def execute(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        limit: int,
    ) -> MessagePack:
        return MessagePack(
            user_id=user_id,
            session_id=session_id,
            messages=tuple(self._stored.messages[-limit:]),
        )


class _Retrieval:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *, message, session_history, limit):
        del message, limit
        self.calls += 1
        pack = MemoryPack(
            query_id=uuid4(),
            user_id=session_history.user_id,
            session_id=session_history.session_id,
            memories=(),
        )
        return TurnMemoryPacks(seeds=pack, expanded=pack)


class _MemoryLearning:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def execute(self, **values):
        self.batch_sizes.append(len(values["target_messages"]))
        return MemoryLearningResult(memory_outcomes=(), errors=())


class LearnFromMessagesTests(unittest.TestCase):
    def test_persists_and_learns_in_one_defined_batch_size(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        stored = _StoreMessage()
        retrieval = _Retrieval()
        learning = _MemoryLearning()
        service = LearnFromMessages(
            get_session_history=_History(stored),
            retriever=retrieval,
            store_message=stored,
            memory_learning=learning,
            maximum_batch_messages=6,
        )
        messages = tuple(
            Message(
                message_id=uuid4(),
                session_id=session_id,
                role="user" if index % 2 == 0 else "assistant",
                agent_id="Alice" if index % 2 == 0 else "Bob",
                content=f"message {index}",
                created_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
            )
            for index in range(7)
        )

        result = service.execute(user_id=user_id, messages=messages)

        self.assertEqual(result.messages, messages)
        self.assertEqual(stored.messages, list(messages))
        self.assertEqual(learning.batch_sizes, [6, 1])
        self.assertEqual(retrieval.calls, 2)


if __name__ == "__main__":
    unittest.main()
