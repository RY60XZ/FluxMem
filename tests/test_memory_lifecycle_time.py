from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fluxmem.application.lifecycle import LifecycleAssigner
from fluxmem.application.llm.usage import ModelUsageCollector
from fluxmem.application.memory_learning import MemoryLearning
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.llm import ProposedMemory
from fluxmem.domain.message import Message


class _Extractor:
    def extract(self, **values):
        source = values["target_messages"][0]
        return (
            ProposedMemory(
                content="Alice lives in Toronto.",
                source_message_id=source.message_id,
                session_applicability=None,
            ),
        )


class _StoreMemory:
    def __init__(self) -> None:
        self.values: dict[str, object] | None = None

    def execute(self, **values) -> UUID:
        self.values = values
        return values["memory"].memory_id


class MemoryLifecycleTimeTests(unittest.TestCase):
    def test_lifecycle_anchor_uses_source_event_time(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        source_time = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
        source = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="user",
            agent_id="Alice",
            content="I live in Toronto.",
            created_at=source_time,
        )
        memory_pack = MemoryPack(
            query_id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            memories=(),
        )
        store = _StoreMemory()
        learning = MemoryLearning(
            store_memory=store,
            memory_extractor=_Extractor(),
            lifecycle_assigner=LifecycleAssigner(),
        )

        learning.execute(
            user_id=user_id,
            session_history=MessagePack(
                user_id=user_id,
                session_id=session_id,
                messages=(source,),
            ),
            target_messages=(source,),
            extraction_pack=memory_pack,
            usage_collector=ModelUsageCollector(),
            diagnostics_collector=None,
        )

        assert store.values is not None
        lifecycle = store.values["lifecycle"]
        self.assertEqual(lifecycle.retention_anchor, source_time)
        self.assertEqual(store.values["indexed_at"], source_time)
        self.assertLess(
            lifecycle.retention_at(source_time + timedelta(days=30)),
            lifecycle.retention_snapshot,
        )


if __name__ == "__main__":
    unittest.main()
