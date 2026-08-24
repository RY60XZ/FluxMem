from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite
from uuid import UUID


EMBEDDING_DIMENSIONS = 1536


class IndexStatus(StrEnum):
    READY = "ready"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class Embedding:
    """One provider-produced vector and the model space it belongs to."""

    values: tuple[float, ...]
    model: str

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("embedding model cannot be blank")
        if not self.values:
            raise ValueError("embedding cannot be empty")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("embedding values must be finite")
        if not any(value != 0.0 for value in self.values):
            raise ValueError("embedding cannot be the zero vector")


@dataclass(frozen=True, slots=True)
class MemoryIndex:
    """Database-independent input for one local retrieval projection."""

    memory_id: UUID
    text: str
    embedding: Embedding | None
    status: IndexStatus
    indexed_at: datetime

    def __post_init__(self) -> None:
        if self.status is IndexStatus.READY and self.embedding is None:
            raise ValueError("a ready projection requires an embedding")
        if self.status is IndexStatus.PENDING and self.embedding is not None:
            raise ValueError("a pending projection cannot contain an embedding")


@dataclass(frozen=True, slots=True)
class MemorySearchQuery:
    """A scoped, bounded hybrid-search request for the PostgreSQL adapter."""

    user_id: UUID
    session_id: UUID
    text: str
    embedding: Embedding | None
    as_of: datetime
    limit: int
    candidate_limit: int
    rrf_k: int
    dense_weight: float
    lexical_weight: float
    relevance_floor: float

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("retrieval limit must be positive")
        if self.candidate_limit < self.limit:
            raise ValueError("candidate limit cannot be smaller than result limit")
        if self.rrf_k < 1:
            raise ValueError("RRF constant must be positive")
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.dense_weight == 0 and self.lexical_weight == 0:
            raise ValueError("at least one retrieval source must be weighted")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise ValueError("relevance floor must be between 0 and 1")
