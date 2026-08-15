from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: UUID
    message_id: UUID
    content: str
    created_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    session_scope : UUID | None = None