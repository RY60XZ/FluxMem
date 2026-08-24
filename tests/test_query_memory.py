from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem.application.query_memory import QueryMemory
from fluxmem.domain.info_pack import MemoryPack, MessagePack, TurnMemoryPacks
from fluxmem.domain.llm import GeneratedAnswer
from fluxmem.domain.message import Message


class _History:
    def __init__(self, pack: MessagePack) -> None:
        self.pack = pack

    def execute(self, **values):
        del values
        return self.pack


class _Retrieval:
    def __init__(self, packs: TurnMemoryPacks) -> None:
        self.packs = packs

    def execute(self, **values):
        del values
        return self.packs


class _AnswerGenerator:
    def generate(self, **values):
        del values
        return GeneratedAnswer(content="Alice spoke.", context_memory_ids=())


class QueryMemoryTests(unittest.TestCase):
    def test_returns_answer_without_a_message_store_dependency(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        history = MessagePack(
            user_id=user_id,
            session_id=session_id,
            messages=(
                Message(
                    message_id=uuid4(),
                    session_id=session_id,
                    role="user",
                    agent_id="Alice",
                    content="Hello",
                    created_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
                ),
            ),
        )
        memory_pack = MemoryPack(
            query_id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            memories=(),
        )
        service = QueryMemory(
            get_session_history=_History(history),
            retrieval_for_answering=_Retrieval(
                TurnMemoryPacks(seeds=memory_pack, expanded=memory_pack)
            ),
            answer_generator=_AnswerGenerator(),
        )

        result = service.execute_text(
            user_id=user_id,
            session_id=session_id,
            content="Who spoke?",
            created_at=datetime(2023, 5, 9, tzinfo=timezone.utc),
        )

        self.assertEqual(result.answer.content, "Alice spoke.")
        self.assertEqual(result.query.created_at.day, 9)
        self.assertEqual(result.context_memory_ids, ())


if __name__ == "__main__":
    unittest.main()
