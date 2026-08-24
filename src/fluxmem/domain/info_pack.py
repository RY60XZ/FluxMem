from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from fluxmem.domain.conflict import MemoryConflict
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

    query_id: UUID
    user_id: UUID
    session_id: UUID
    memories: tuple[RetrievedMemory, ...]
    conflicts: tuple[MemoryConflict, ...] = ()
    conflict_expansion_truncated: bool = False


@dataclass(frozen=True, slots=True)
class TurnMemoryPacks:
    """Seed and conflict-expanded views of one persisted retrieval event."""

    seeds: MemoryPack
    expanded: MemoryPack

    def __post_init__(self) -> None:
        if self.seeds.query_id != self.expanded.query_id:
            raise ValueError("turn memory views must share one query ID")
        if self.seeds.user_id != self.expanded.user_id:
            raise ValueError("turn memory views must share one user")
        if self.seeds.session_id != self.expanded.session_id:
            raise ValueError("turn memory views must share one session")
        seed_ids = tuple(
            retrieved.memory.memory_id for retrieved in self.seeds.memories
        )
        expanded_prefix = tuple(
            retrieved.memory.memory_id
            for retrieved in self.expanded.memories[: len(seed_ids)]
        )
        if expanded_prefix != seed_ids:
            raise ValueError("expanded turn context must begin with its seeds")
        if self.seeds.conflicts or self.seeds.conflict_expansion_truncated:
            raise ValueError("seed context cannot contain conflict expansion")


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    """An ephemeral query and the memory context returned for external use."""

    query: Message
    retrieval: TurnMemoryPacks

    def __post_init__(self) -> None:
        if self.query.session_id != self.retrieval.expanded.session_id:
            raise ValueError("query and retrieval must share one session")

    @property
    def query_id(self) -> UUID:
        return self.retrieval.expanded.query_id

    @property
    def context(self) -> MemoryPack:
        return self.retrieval.expanded


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

    query_id: UUID
    user_id: UUID
    session_id: UUID
    used_memories: tuple[MemoryUsage, ...]
