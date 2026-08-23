from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from fluxmem.adapters.postgres.models.retrieval import (
    RetrievalCandidateRow,
    RetrievalQueryRow,
)
from fluxmem.adapters.postgres.repositories.retrievals import (
    SqlAlchemyRetrievalRepository,
)
from fluxmem.domain.info_pack import RetrievedMemory
from fluxmem.domain.memory import Memory
from fluxmem.domain.retrieval import QueryType


class FakeSession:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def add(self, row: object) -> None:
        self.events.append(("add", row))

    def flush(self) -> None:
        self.events.append(("flush", None))

    def add_all(self, rows) -> None:
        self.events.append(("add_all", tuple(rows)))


class RetrievalRepositoryTests(unittest.TestCase):
    def test_flushes_query_before_adding_candidate_rows(self) -> None:
        now = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
        query_id = uuid4()
        session = FakeSession()
        memory = Memory(
            memory_id=uuid4(),
            message_id=uuid4(),
            content="The user prefers coffee.",
            created_at=now,
        )
        candidate = RetrievedMemory(
            memory=memory,
            rank=1,
            score=0.9,
            retention=0.9,
            retrieval_reasons=("dense",),
        )

        repository = SqlAlchemyRetrievalRepository(session)  # type: ignore[arg-type]
        repository.add(
            query_id=query_id,
            session_id=uuid4(),
            query_type=QueryType.ANSWERING,
            candidates=(candidate,),
            created_at=now,
        )

        self.assertEqual(
            [event for event, _ in session.events],
            ["add", "flush", "add_all"],
        )
        self.assertIsInstance(session.events[0][1], RetrievalQueryRow)
        rows = session.events[2][1]
        self.assertIsInstance(rows, tuple)
        self.assertEqual(len(rows), 1)
        self.assertIsInstance(rows[0], RetrievalCandidateRow)
        self.assertEqual(rows[0].query_id, query_id)


if __name__ == "__main__":
    unittest.main()
