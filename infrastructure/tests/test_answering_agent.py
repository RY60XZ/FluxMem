from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fluxmem import (
    MemoryPack,
    MemoryRetrievalResult,
    Message,
    MessageIngestionResult,
    MessagePack,
)
from fluxmem.domain.info_pack import TurnMemoryPacks
from fluxmem_infrastructure.answering import AnsweringAgent
from fluxmem_infrastructure.answering.generator import GeneratedAnswer


class _Memory:
    def __init__(self, *, user_id: UUID, session_id: UUID) -> None:
        pack = MemoryPack(
            query_id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            memories=(),
        )
        self.retrieval = MemoryRetrievalResult(
            query=Message(
                message_id=uuid4(),
                session_id=session_id,
                role="user",
                agent_id=None,
                content="Who spoke?",
                created_at=datetime(2023, 5, 9, tzinfo=timezone.utc),
            ),
            retrieval=TurnMemoryPacks(seeds=pack, expanded=pack),
        )
        self.history = MessagePack(
            user_id=user_id,
            session_id=session_id,
            messages=(),
        )
        self.calls: list[str] = []

    def retrieve(self, **values):
        del values
        self.calls.append("retrieve")
        return self.retrieval

    def get_session_history(self, **values):
        del values
        self.calls.append("history")
        return self.history

    def record_usage(self, **values):
        del values
        self.calls.append("usage")
        return ()

    def ingest_messages(self, **values):
        self.calls.append("ingest")
        return MessageIngestionResult(messages=tuple(values["messages"]))

    def store_messages(self, **values):
        self.calls.append("store")
        return tuple(message.message_id for message in values["messages"])


class _Generator:
    def generate(self, **values):
        del values
        return GeneratedAnswer(
            content="Alice spoke.",
            context_memory_ids=(),
            model="test-model",
            response_id="response-1",
            usage=None,
        )


class AnsweringAgentTests(unittest.TestCase):
    def test_answer_uses_only_memory_retrieval_and_history(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        memory = _Memory(user_id=user_id, session_id=session_id)
        agent = AnsweringAgent(memory=memory, generator=_Generator())

        result = agent.answer(
            user_id=user_id,
            session_id=session_id,
            content="Who spoke?",
        )

        self.assertEqual(result.answer.content, "Alice spoke.")
        self.assertEqual(memory.calls, ["retrieve", "history"])

    def test_run_turn_explicitly_composes_memory_mutations(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        memory = _Memory(user_id=user_id, session_id=session_id)
        agent = AnsweringAgent(memory=memory, generator=_Generator())

        agent.run_turn(
            user_id=user_id,
            session_id=session_id,
            content="Who spoke?",
        )

        self.assertEqual(
            memory.calls,
            ["retrieve", "history", "usage", "ingest", "store"],
        )


if __name__ == "__main__":
    unittest.main()
