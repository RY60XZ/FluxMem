from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.memory_learning import (
    LearnFromMessages,
    MemoryLearning,
    MemoryLearningResult,
)
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.llm import MemoryWriteStatus, ProposedMemory
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
        return pack


class _MemoryLearning:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def execute(self, **values):
        self.batch_sizes.append(len(values["target_messages"]))
        return MemoryLearningResult(memory_outcomes=(), errors=())


class _CandidateExtractor:
    def __init__(self, candidates: tuple[ProposedMemory, ...]) -> None:
        self._candidates = candidates

    def extract(self, **values):
        del values
        return self._candidates


class _CapturingMemoryStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, **values) -> UUID:
        self.calls.append(values)
        return values["memory"].memory_id


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


class MemoryLearningWritePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.source = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id="Alice",
            content="I prefer tea.",
            created_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
        )

    def _execute(
        self,
        *,
        candidates: tuple[ProposedMemory, ...],
    ) -> tuple[MemoryLearningResult, _CapturingMemoryStore]:
        pack = MemoryPack(
            query_id=uuid4(),
            user_id=self.user_id,
            session_id=self.session_id,
            memories=(),
        )
        store = _CapturingMemoryStore()
        learning = MemoryLearning(
            store_memory=store,
            memory_extractor=_CandidateExtractor(candidates),
            lifecycle_assigner=LifecycleAssigner(),
        )
        result = learning.execute(
            user_id=self.user_id,
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=(self.source,),
            ),
            target_messages=(self.source,),
            extraction_pack=pack,
            usage_collector=ModelUsageCollector(),
            diagnostics_collector=None,
        )
        return result, store

    def _candidate(self) -> ProposedMemory:
        return ProposedMemory(
            content="Alice prefers tea.",
            source_message_id=self.source.message_id,
            session_applicability=None,
        )

    def test_stores_duplicate_extraction_candidates_independently(self) -> None:
        candidate = self._candidate()

        result, store = self._execute(
            candidates=(candidate, candidate),
        )

        self.assertEqual(len(store.calls), 2)
        self.assertEqual(
            tuple(outcome.status for outcome in result.memory_outcomes),
            (MemoryWriteStatus.STORED, MemoryWriteStatus.STORED),
        )
        self.assertNotEqual(
            result.memory_outcomes[0].memory_id,
            result.memory_outcomes[1].memory_id,
        )


if __name__ == "__main__":
    unittest.main()
