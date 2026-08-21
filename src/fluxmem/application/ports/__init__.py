"""Interfaces implemented by persistence and provider adapters."""

from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

__all__ = ("EmbeddingProvider", "EmbeddingProviderError")
