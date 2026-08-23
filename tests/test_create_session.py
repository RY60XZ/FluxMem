from __future__ import annotations

import unittest
from uuid import uuid4

from fluxmem.application.write.create_session import CreateSession


class FakeSessions:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.created = None

    def ensure_user(self, *, user_id) -> None:
        del user_id
        self.events.append("user")

    def add(self, *, session) -> None:
        self.events.append("session")
        self.created = session


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.sessions = FakeSessions(self.events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def flush(self) -> None:
        self.events.append("flush")

    def commit(self) -> None:
        self.events.append("commit")


class CreateSessionTests(unittest.TestCase):
    def test_flushes_new_user_before_inserting_owned_session(self) -> None:
        unit_of_work = FakeUnitOfWork()
        service = CreateSession(unit_of_work_factory=lambda: unit_of_work)
        user_id = uuid4()

        created = service.execute(user_id=user_id)

        self.assertEqual(created.user_id, user_id)
        self.assertIs(unit_of_work.sessions.created, created)
        self.assertEqual(
            unit_of_work.events,
            ["user", "flush", "session", "commit"],
        )


if __name__ == "__main__":
    unittest.main()
