from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


@dataclass(frozen=True, slots=True)
class RetrievalCandidateDiagnostics:
    """First-stage score components for one returned memory."""

    initial_rank: int
    fusion_score: float
    lifecycle_multiplier: float
    dense_rank: int | None = None
    dense_similarity: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalQueryDiagnostics:
    """The exact search input and bounds used for one retrieval."""

    search_text: str
    message_count: int
    result_limit: int
    source_candidate_limit: int
    returned_count: int
    dense_enabled: bool
    lexical_enabled: bool
    embedding_model: str | None = None
    embedding_error: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """A memory together with metadata produced by one retrieval query."""

    memory: Memory
    rank: int
    score: float
    retention: float
    retrieval_reasons: tuple[str, ...]
    diagnostics: RetrievalCandidateDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class MemoryPack:
    """The ordered result of one scoped memory-retrieval query."""

    query_id: UUID
    user_id: UUID
    session_id: UUID
    memories: tuple[RetrievedMemory, ...]
    diagnostics: RetrievalQueryDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    """An ephemeral query and the memory context returned for external use."""

    query: Message
    context: MemoryPack

    def __post_init__(self) -> None:
        if self.query.session_id != self.context.session_id:
            raise ValueError("query and retrieval must share one session")

    @property
    def query_id(self) -> UUID:
        return self.context.query_id


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
