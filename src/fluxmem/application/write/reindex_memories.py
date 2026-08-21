from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fluxmem.application.ports.clock import Clock, SystemClock
from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from fluxmem.application.ports.unit_of_work import UnitOfWork
from fluxmem.domain.retrieval import EMBEDDING_DIMENSIONS


@dataclass(frozen=True, slots=True)
class ReindexSummary:
    attempted: int
    indexed: int
    failed: int


class ReindexPendingMemories:
    """Repair a bounded batch of lexical-only memory projections."""

    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], UnitOfWork],
        embedding_provider: EmbeddingProvider,
        clock: Clock | None = None,
    ) -> None:
        if embedding_provider.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "embedding provider dimensions do not match the database schema"
            )
        self._unit_of_work_factory = unit_of_work_factory
        self._embedding_provider = embedding_provider
        self._clock = clock or SystemClock()

    def execute(self, *, limit: int = 100) -> ReindexSummary:
        if limit < 1:
            raise ValueError("reindex limit must be positive")

        with self._unit_of_work_factory() as unit_of_work:
            pending = unit_of_work.memory_indexes.list_pending(limit=limit)

        indexed = 0
        failed = 0
        for memory in pending:
            try:
                embedding = self._embedding_provider.embed(text=memory.content)
            except EmbeddingProviderError:
                failed += 1
                continue
            if len(embedding.values) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    "embedding provider returned a vector with invalid dimensions"
                )

            with self._unit_of_work_factory() as unit_of_work:
                updated = unit_of_work.memory_indexes.mark_ready(
                    memory_id=memory.memory_id,
                    embedding=embedding,
                    indexed_at=self._clock.now(),
                )
                unit_of_work.commit()
            if updated:
                indexed += 1

        return ReindexSummary(
            attempted=len(pending),
            indexed=indexed,
            failed=failed,
        )
