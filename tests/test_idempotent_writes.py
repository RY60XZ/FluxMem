from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from fluxmem.application.write.create_session import CreateSession
from fluxmem.application.write.store_message import StoreMessage
from fluxmem.domain.message import Message, Session


class _Sessions:
    def __init__(self) -> None:
        self.users: set[UUID] = set()
        self.sessions: dict[UUID, Session] = {}

    def ensure_user(self, *, user_id: UUID) -> None:
        self.users.add(user_id)

    def add(self, *, session: Session) -> None:
        if session.session_id in self.sessions:
            raise AssertionError("duplicate session write")
        self.sessions[session.session_id] = session

    def is_owned_by(self, *, session_id: UUID, user_id: UUID) -> bool:
        session = self.sessions.get(session_id)
        return session is not None and session.user_id == user_id


class _Messages:
    def __init__(self, sessions: _Sessions) -> None:
        self._sessions = sessions
        self.messages: dict[UUID, Message] = {}

    def get(self, *, message_id: UUID, user_id: UUID) -> Message | None:
        message = self.messages.get(message_id)
        if message is None:
            return None
        if not self._sessions.is_owned_by(
            session_id=message.session_id,
            user_id=user_id,
        ):
            return None
        return message

    def add(self, *, message: Message) -> None:
        if message.message_id in self.messages:
            raise AssertionError("duplicate message write")
        self.messages[message.message_id] = message


class _UnitOfWork:
    def __init__(self) -> None:
        self.sessions = _Sessions()
        self.messages = _Messages(self.sessions)
        self.commit_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.commit_count += 1


class IdempotentWriteTests(unittest.TestCase):
    def test_replaying_session_and_message_is_idempotent(self) -> None:
        unit_of_work = _UnitOfWork()

        def factory() -> _UnitOfWork:
            return unit_of_work

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        session_id = UUID("00000000-0000-0000-0000-000000000002")
        message = Message(
            message_id=UUID("00000000-0000-0000-0000-000000000003"),
            session_id=session_id,
            role="user",
            agent_id="Alice",
            content="Alice: I live in Toronto.",
            created_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
        )

        sessions = CreateSession(unit_of_work_factory=factory)
        messages = StoreMessage(unit_of_work_factory=factory)
        sessions.execute(user_id=user_id, session_id=session_id)
        sessions.execute(user_id=user_id, session_id=session_id)
        messages.execute(user_id=user_id, message=message)
        messages.execute(user_id=user_id, message=message)

        self.assertEqual(len(unit_of_work.sessions.sessions), 1)
        self.assertEqual(len(unit_of_work.messages.messages), 1)
        self.assertEqual(unit_of_work.commit_count, 2)

    def test_replaying_changed_message_is_rejected(self) -> None:
        unit_of_work = _UnitOfWork()

        def factory() -> _UnitOfWork:
            return unit_of_work

        user_id = UUID("00000000-0000-0000-0000-000000000001")
        session_id = UUID("00000000-0000-0000-0000-000000000002")
        message_id = UUID("00000000-0000-0000-0000-000000000003")
        sessions = CreateSession(unit_of_work_factory=factory)
        messages = StoreMessage(unit_of_work_factory=factory)
        sessions.execute(user_id=user_id, session_id=session_id)
        original = Message(
            message_id=message_id,
            session_id=session_id,
            role="user",
            agent_id=None,
            content="Original",
            created_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
        )
        changed = Message(
            message_id=message_id,
            session_id=session_id,
            role="user",
            agent_id=None,
            content="Changed",
            created_at=original.created_at,
        )
        messages.execute(user_id=user_id, message=original)

        with self.assertRaisesRegex(ValueError, "different content"):
            messages.execute(user_id=user_id, message=changed)


if __name__ == "__main__":
    unittest.main()
