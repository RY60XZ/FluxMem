from __future__ import annotations

from typing import Protocol

from fluxmem.domain.retrieval import Embedding


class EmbeddingProviderError(RuntimeError):
    """A temporary provider failure for which lexical fallback is allowed."""


class EmbeddingProvider(Protocol):
    """Generate embeddings in one fixed-dimensional model space."""

    dimensions: int

    def embed(self, *, text: str) -> Embedding: ...
