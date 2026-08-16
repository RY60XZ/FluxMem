from __future__ import annotations

from typing import Protocol
from uuid import UUID

from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


class MessageRepository(Protocol):
    def get(self, *, message_id: UUID, user_id: UUID) -> Message | None: ...


class SessionRepository(Protocol):
    def is_owned_by(self, *, session_id: UUID, user_id: UUID) -> bool: ...


class MemoryRepository(Protocol):
    def add(self, *, memory: Memory) -> None: ...
