from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """A memory together with metadata produced by one retrieval query."""

    memory: Memory
    rank: int
    score: float
    retention: float
    retrieval_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryPack:
    """The ordered result of one scoped memory-retrieval query."""

    user_id: UUID
    session_id: UUID
    memories: tuple[RetrievedMemory, ...]


@dataclass(frozen=True, slots=True)
class MessagePack:
    """Chronological history for one user-owned session."""

    user_id: UUID
    session_id: UUID
    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        if any(message.session_id != self.session_id for message in self.messages):
            raise ValueError("every message must belong to the pack's session")


class UsageType(StrEnum):
    CONTEXT_INCLUDED = "context_included"
    MODEL_ATTRIBUTED = "model_attributed"


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """One memory that was actually used after retrieval."""

    memory_id: UUID
    usage_type: UsageType
    rank: int | None = None
    contribution: float | None = None


@dataclass(frozen=True, slots=True)
class FeedbackPack:
    """Usage observations for a completed retrieval query."""

    user_id: UUID
    session_id: UUID
    used_memories: tuple[MemoryUsage, ...]
