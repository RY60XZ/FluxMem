from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fluxmem.application.ports.embeddings import EmbeddingProviderError
from fluxmem.application.read.retrieval import (
    HybridRetrievalSettings,
    RetrievalForAnswering,
)
from fluxmem.domain.info_pack import MessagePack
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import EMBEDDING_DIMENSIONS, Embedding, QueryType


@dataclass
class FixedClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


class FakeEmbeddingProvider:
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.texts: list[str] = []

    def embed(self, *, text: str) -> Embedding:
        self.texts.append(text)
        if self.fails:
            raise EmbeddingProviderError("provider unavailable")
        return Embedding(
            values=(1.0,) + (0.0,) * (EMBEDDING_DIMENSIONS - 1),
            model="test-embedding-v1",
        )


class FakeSessions:
    def is_owned_by(self, *, session_id: UUID, user_id: UUID) -> bool:
        return True


class FakeMemories:
    def __init__(self) -> None:
        self.query = None

    def search(self, *, query):
        self.query = query
        return ()


class FakeRetrievals:
    def __init__(self) -> None:
        self.added = None

    def add(self, **values) -> None:
        self.added = values


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.sessions = FakeSessions()
        self.memories = FakeMemories()
        self.retrievals = FakeRetrievals()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def commit(self) -> None:
        self.committed = True


class HybridRetrievalApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user_id = uuid4()
        self.session_id = uuid4()
        self.query_id = uuid4()
        self.now = datetime(2026, 8, 21, 12, tzinfo=timezone.utc)
        self.uow = FakeUnitOfWork()

    def test_builds_bounded_hybrid_query_and_persists_empty_result(self) -> None:
        provider = FakeEmbeddingProvider()
        service = RetrievalForAnswering(
            unit_of_work_factory=lambda: self.uow,
            embedding_provider=provider,
            settings=HybridRetrievalSettings(
                max_query_messages=2,
                max_query_characters=10,
                minimum_candidate_limit=2,
                maximum_candidate_limit=20,
            ),
            clock=FixedClock(self.now),
            query_id_factory=lambda: self.query_id,
        )
        old_message = self._message("old context")
        recent_message = self._message("recent context")
        current_message = self._message("question")

        result = service.execute(
            message=current_message,
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=(old_message, recent_message),
            ),
            limit=2,
        )

        self.assertEqual(result.query_id, self.query_id)
        self.assertEqual(result.memories, ())
        self.assertTrue(self.uow.committed)
        self.assertEqual(provider.texts, ["r\nquestion"])
        self.assertEqual(self.uow.memories.query.text, "r\nquestion")
        self.assertIsNotNone(self.uow.memories.query.embedding)
        self.assertEqual(self.uow.memories.query.candidate_limit, 10)
        self.assertEqual(
            self.uow.retrievals.added["query_type"],
            QueryType.ANSWERING,
        )
        self.assertEqual(self.uow.retrievals.added["candidates"], ())

    def test_embedding_provider_failure_falls_back_to_lexical(self) -> None:
        provider = FakeEmbeddingProvider(fails=True)
        service = RetrievalForAnswering(
            unit_of_work_factory=lambda: self.uow,
            embedding_provider=provider,
            clock=FixedClock(self.now),
            query_id_factory=lambda: self.query_id,
        )
        message = self._message("volcano")

        service.execute(
            message=message,
            session_history=MessagePack(
                user_id=self.user_id,
                session_id=self.session_id,
                messages=(),
            ),
        )

        self.assertIsNone(self.uow.memories.query.embedding)
        self.assertEqual(self.uow.memories.query.text, "volcano")
        self.assertTrue(self.uow.committed)

    def _message(self, content: str) -> Message:
        return Message(
            message_id=uuid4(),
            session_id=self.session_id,
            role="user",
            agent_id=None,
            content=content,
            created_at=self.now,
        )


if __name__ == "__main__":
    unittest.main()
