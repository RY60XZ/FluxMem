from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from fluxmem.adapters.postgres.repositories.conflicts import (
    SqlAlchemyConflictRepository,
)
from fluxmem.adapters.postgres.repositories.memories import (
    SqlAlchemyMemoryRepository,
)
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


class TemporalRetrievalTests(unittest.TestCase):
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

    def test_conflict_expansion_does_not_filter_by_validity_interval(self) -> None:
        session = _CapturingSession()
        repository = SqlAlchemyConflictRepository(session)
        repository.expand(
            seed_memory_ids=(uuid4(),),
            user_id=uuid4(),
            session_id=uuid4(),
            as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
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
