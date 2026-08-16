from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass(frozen=True, slots=True)
class User:
    user_id: UUID

@dataclass(frozen=True, slots=True)
class Session:
    session_id: UUID
    user_id: UUID

@dataclass(frozen=True, slots=True)
class Message:
    message_id: UUID
    session_id: UUID
    role: str
    agent_id: str | None
    content: str
    created_at: datetime
