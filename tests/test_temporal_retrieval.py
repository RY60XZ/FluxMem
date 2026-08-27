from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from fluxmem.adapters.postgres.repositories.memories import (
    SqlAlchemyMemoryRepository,
)
from fluxmem.application.read.retrieval import (
    HybridMemoryRetriever,
    HybridRetrievalSettings,
)
from fluxmem.domain.info_pack import MessagePack
from fluxmem.domain.message import Message
from fluxmem.domain.retrieval import MemorySearchQuery


class _EmptyResult:
    def all(self) -> list[object]:
        return []


class _CapturingSession:
    def __init__(self) -> None:
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return _EmptyResult()


class _OwnedSessions:
    def is_owned_by(self, *, session_id, user_id) -> bool:
        del session_id, user_id
        return True


class _CapturingMemories:
    def __init__(self) -> None:
        self.query: MemorySearchQuery | None = None

    def search(self, *, query: MemorySearchQuery):
        self.query = query
        return ()


class _CapturingRetrievals:
    def __init__(self) -> None:
        self.created_at: datetime | None = None

    def add(self, *, created_at, **values) -> None:
        del values
        self.created_at = created_at


class _RetrievalUnitOfWork:
    def __init__(self) -> None:
        self.sessions = _OwnedSessions()
        self.memories = _CapturingMemories()
        self.retrievals = _CapturingRetrievals()
        self.commit_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback

    def commit(self) -> None:
        self.commit_count += 1


class TemporalRetrievalTests(unittest.TestCase):
    def test_question_only_profile_excludes_stored_conversation_text(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        query_time = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
        prior = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="assistant",
            agent_id="Alice",
            content="Unrelated previous conversation.",
            created_at=query_time,
        )
        query = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="user",
            agent_id=None,
            content="Where does Alice live?",
            created_at=query_time,
        )
        unit_of_work = _RetrievalUnitOfWork()
        retriever = HybridMemoryRetriever(
            unit_of_work_factory=lambda: unit_of_work,
            settings=HybridRetrievalSettings(max_query_messages=1),
        )

        result = retriever.execute(
            message=query,
            session_history=MessagePack(
                user_id=user_id,
                session_id=session_id,
                messages=(prior,),
            ),
            limit=50,
        )

        assert unit_of_work.memories.query is not None
        self.assertEqual(
            unit_of_work.memories.query.text,
            "Where does Alice live?",
        )
        assert result.diagnostics is not None
        self.assertEqual(result.diagnostics.search_text, query.content)
        self.assertEqual(result.diagnostics.message_count, 1)
        self.assertEqual(result.diagnostics.result_limit, 50)
        self.assertEqual(result.diagnostics.returned_count, 0)

    def test_hybrid_retrieval_uses_the_query_message_time(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        query_time = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
        query = Message(
            message_id=uuid4(),
            session_id=session_id,
            role="user",
            agent_id=None,
            content="Where does Alice live?",
            created_at=query_time,
        )
        unit_of_work = _RetrievalUnitOfWork()
        retriever = HybridMemoryRetriever(
            unit_of_work_factory=lambda: unit_of_work,
        )

        retriever.execute(
            message=query,
            session_history=MessagePack(
                user_id=user_id,
                session_id=session_id,
                messages=(),
            ),
        )

        assert unit_of_work.memories.query is not None
        self.assertEqual(unit_of_work.memories.query.as_of, query_time)
        self.assertEqual(unit_of_work.retrievals.created_at, query_time)
        self.assertEqual(unit_of_work.commit_count, 1)

    def test_memory_search_does_not_filter_by_validity_interval(self) -> None:
        session = _CapturingSession()
        repository = SqlAlchemyMemoryRepository(session)
        repository.search(
            query=MemorySearchQuery(
                user_id=uuid4(),
                session_id=uuid4(),
                text="past event",
                embedding=None,
                as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
                limit=10,
                candidate_limit=50,
                rrf_k=60,
                dense_weight=0.0,
                lexical_weight=1.0,
                relevance_floor=0.6,
            )
        )

        sql = _compiled_sql(session.statement)
        self.assertNotIn("memories.valid_from <=", sql)
        self.assertNotIn("memories.valid_to >=", sql)


def _compiled_sql(statement) -> str:
    if statement is None:
        raise AssertionError("repository did not execute a statement")
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
        )
    )


if __name__ == "__main__":
    unittest.main()
