from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem.domain.conflict import ConflictProposal
from fluxmem.application.ports.embeddings import EmbeddingProviderError
from fluxmem.application.write.store_memory import StoreMemory
from fluxmem.domain.memory import Memory
from fluxmem.domain.lifecycle import DecisionSource, LifecycleDecision, Tier
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    Embedding,
    IndexStatus,
    QueryType,
)


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


class FakeEmbeddingProvider:
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails

    def embed(self, *, text: str) -> Embedding:
        if self.fails:
            raise EmbeddingProviderError("temporary failure")
        return Embedding(
            values=(1.0,) + (0.0,) * (EMBEDDING_DIMENSIONS - 1),
            model="test-embedding-v1",
        )


class FakeMessages:
    def __init__(self, message: Message) -> None:
        self.message = message

    def get(self, *, message_id, user_id):
        return self.message if message_id == self.message.message_id else None


class RecordingRepository:
    def __init__(self) -> None:
        self.added = []

    def add(self, **values) -> None:
        self.added.append(values)


class FakeRetrievals:
    def __init__(self, candidate_ids=()) -> None:
        self.candidate_ids = candidate_ids
        self.request = None

    def candidate_ids_for_query(self, **values):
        self.request = values
        return self.candidate_ids


class FakeUnitOfWork:
    def __init__(self, message: Message, *, candidate_ids=()) -> None:
        self.messages = FakeMessages(message)
        self.memories = RecordingRepository()
        self.memory_indexes = RecordingRepository()
        self.lifecycles = RecordingRepository()
        self.conflicts = RecordingRepository()
        self.retrievals = FakeRetrievals(candidate_ids)
        self.committed = False
        self.active = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.active = False

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.committed = True


class InspectingLifecycleEvaluator:
    def __init__(self, unit_of_work: FakeUnitOfWork) -> None:
        self.unit_of_work = unit_of_work
        self.called_while_transaction_open = None

    def evaluate(self, **values) -> LifecycleDecision:
        del values
        self.called_while_transaction_open = self.unit_of_work.active
        return LifecycleDecision(
            importance=0.7,
            tier=Tier.SHORT_TERM,
            initial_retention=0.7,
            reason_codes=("test",),
            confidence=0.8,
            decision_source=DecisionSource.LLM_PRIMARY,
        )


class FailingLifecycleEvaluator:
    def evaluate(self, **values):
        del values
        raise RuntimeError("provider unavailable")


class StoreMemoryIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 21, tzinfo=timezone.utc)
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.message = Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id=None,
            content="Remember volcanoes",
            created_at=self.now,
        )
        self.memory = Memory(
            memory_id=uuid4(),
            message_id=self.message.message_id,
            content="The user likes volcanoes",
            created_at=self.now,
        )

    def test_successful_embedding_creates_ready_projection(self) -> None:
        unit_of_work = FakeUnitOfWork(self.message)
        service = StoreMemory(
            unit_of_work_factory=lambda: unit_of_work,
            embedding_provider=FakeEmbeddingProvider(),
            clock=FixedClock(self.now),
        )

        service.execute(user_id=self.user_id, memory=self.memory)

        index = unit_of_work.memory_indexes.added[0]["index"]
        self.assertEqual(index.status, IndexStatus.READY)
        self.assertIsNotNone(index.embedding)
        self.assertTrue(unit_of_work.committed)

    def test_temporary_embedding_failure_creates_pending_projection(self) -> None:
        unit_of_work = FakeUnitOfWork(self.message)
        service = StoreMemory(
            unit_of_work_factory=lambda: unit_of_work,
            embedding_provider=FakeEmbeddingProvider(fails=True),
            clock=FixedClock(self.now),
        )

        service.execute(user_id=self.user_id, memory=self.memory)

        index = unit_of_work.memory_indexes.added[0]["index"]
        self.assertEqual(index.status, IndexStatus.PENDING)
        self.assertIsNone(index.embedding)
        self.assertTrue(unit_of_work.committed)

    def test_only_turn_retrieval_candidates_become_conflict_edges(self) -> None:
        eligible_id = uuid4()
        fabricated_id = uuid4()
        query_id = uuid4()
        unit_of_work = FakeUnitOfWork(
            self.message,
            candidate_ids=(eligible_id,),
        )
        service = StoreMemory(
            unit_of_work_factory=lambda: unit_of_work,
            embedding_provider=FakeEmbeddingProvider(),
            clock=FixedClock(self.now),
        )

        service.execute(
            user_id=self.user_id,
            memory=self.memory,
            write_context_query_id=query_id,
            conflict_proposals=(
                ConflictProposal(eligible_id, confidence=0.9),
                ConflictProposal(eligible_id, confidence=0.2),
                ConflictProposal(fabricated_id, confidence=0.8),
                ConflictProposal(self.memory.memory_id, confidence=0.7),
            ),
        )

        self.assertEqual(len(unit_of_work.conflicts.added), 1)
        conflict = unit_of_work.conflicts.added[0]["conflict"]
        self.assertEqual(
            {conflict.memory_a_id, conflict.memory_b_id},
            {self.memory.memory_id, eligible_id},
        )
        self.assertEqual(conflict.confidence, 0.9)
        self.assertEqual(
            unit_of_work.retrievals.request,
            {
                "query_id": query_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "query_type": QueryType.ANSWERING,
            },
        )

    def test_lifecycle_provider_runs_outside_database_context(self) -> None:
        unit_of_work = FakeUnitOfWork(self.message)
        evaluator = InspectingLifecycleEvaluator(unit_of_work)
        service = StoreMemory(
            unit_of_work_factory=lambda: unit_of_work,
            lifecycle_evaluator=evaluator,
            clock=FixedClock(self.now),
        )

        service.execute(user_id=self.user_id, memory=self.memory)

        self.assertFalse(evaluator.called_while_transaction_open)
        self.assertTrue(unit_of_work.committed)

    def test_lifecycle_provider_failure_uses_rule_fallback(self) -> None:
        unit_of_work = FakeUnitOfWork(self.message)
        service = StoreMemory(
            unit_of_work_factory=lambda: unit_of_work,
            lifecycle_evaluator=FailingLifecycleEvaluator(),
            clock=FixedClock(self.now),
        )

        service.execute(user_id=self.user_id, memory=self.memory)

        lifecycle = unit_of_work.lifecycles.added[0]["lifecycle"]
        self.assertIs(
            lifecycle.decision_source,
            DecisionSource.RULES_FALLBACK,
        )


if __name__ == "__main__":
    unittest.main()
