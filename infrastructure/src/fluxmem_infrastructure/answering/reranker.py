from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from fluxmem import MemoryPack, Message, ModelTokenUsage


@dataclass(frozen=True, slots=True)
class MemoryRerankerSettings:
    model: str
    timeout_seconds: float = 60.0
    maximum_candidates: int = 50

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("reranker model cannot be blank")
        if self.timeout_seconds <= 0 or self.maximum_candidates < 1:
            raise ValueError("reranker limits must be positive")


@dataclass(frozen=True, slots=True)
class RerankProviderResult:
    index: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankProviderResponse:
    results: tuple[RerankProviderResult, ...]
    model: str
    response_id: str | None = None
    provider: str | None = None
    total_tokens: int | None = None
    search_units: int | None = None


class RerankProvider(Protocol):
    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: tuple[str, ...],
        top_n: int,
        timeout_seconds: float,
    ) -> RerankProviderResponse: ...


@dataclass(frozen=True, slots=True)
class RerankerCandidateDiagnostics:
    memory_id: UUID
    initial_rank: int
    final_rank: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankerDiagnostics:
    status: str
    attempted: bool
    candidate_count: int
    model: str | None
    response_id: str | None
    usage: ModelTokenUsage | None
    error: str | None = None
    provider: str | None = None
    search_units: int | None = None
    candidates: tuple[RerankerCandidateDiagnostics, ...] = ()


@dataclass(frozen=True, slots=True)
class RerankingOutcome:
    memory_pack: MemoryPack
    diagnostics: RerankerDiagnostics


class MemoryReranker(Protocol):
    def rerank(
        self,
        *,
        query: Message,
        memory_pack: MemoryPack,
    ) -> RerankingOutcome: ...


class ModelMemoryReranker:
    """Reorder a bounded memory pack through a native rerank provider."""

    def __init__(
        self,
        *,
        provider: RerankProvider,
        settings: MemoryRerankerSettings,
    ) -> None:
        self._provider = provider
        self._settings = settings

    def rerank(
        self,
        *,
        query: Message,
        memory_pack: MemoryPack,
    ) -> RerankingOutcome:
        candidates = memory_pack.memories
        if not candidates:
            return RerankingOutcome(
                memory_pack=memory_pack,
                diagnostics=RerankerDiagnostics(
                    status="skipped",
                    attempted=False,
                    candidate_count=0,
                    model=None,
                    response_id=None,
                    usage=None,
                ),
            )
        if len(candidates) > self._settings.maximum_candidates:
            return self._failed(
                memory_pack=memory_pack,
                attempted=False,
                error=(
                    "candidate count exceeds reranker maximum: "
                    f"{len(candidates)} > {self._settings.maximum_candidates}"
                ),
            )

        try:
            response = self._provider.rerank(
                model=self._settings.model,
                query=query.content,
                documents=tuple(
                    _reranker_document(retrieved.memory)
                    for retrieved in candidates
                ),
                top_n=len(candidates),
                timeout_seconds=self._settings.timeout_seconds,
            )
        except Exception as error:
            return self._failed(
                memory_pack=memory_pack,
                attempted=True,
                error=_error_text(error),
            )

        try:
            _validate_results(
                results=response.results,
                candidate_count=len(candidates),
            )
        except Exception as error:
            return self._failed(
                memory_pack=memory_pack,
                attempted=True,
                error=_error_text(error),
                response=response,
            )

        reranked = tuple(
            replace(
                candidates[result.index],
                rank=rank,
                retrieval_reasons=tuple(
                    dict.fromkeys(
                        (
                            *candidates[result.index].retrieval_reasons,
                            "reranker",
                        )
                    )
                ),
            )
            for rank, result in enumerate(response.results, start=1)
        )
        candidate_diagnostics = tuple(
            RerankerCandidateDiagnostics(
                memory_id=candidates[result.index].memory.memory_id,
                initial_rank=candidates[result.index].rank,
                final_rank=rank,
                relevance_score=result.relevance_score,
            )
            for rank, result in enumerate(response.results, start=1)
        )
        return RerankingOutcome(
            memory_pack=replace(memory_pack, memories=reranked),
            diagnostics=RerankerDiagnostics(
                status="completed",
                attempted=True,
                candidate_count=len(candidates),
                model=response.model or self._settings.model,
                response_id=response.response_id,
                usage=_token_usage(response.total_tokens),
                provider=response.provider,
                search_units=response.search_units,
                candidates=candidate_diagnostics,
            ),
        )

    def _failed(
        self,
        *,
        memory_pack: MemoryPack,
        attempted: bool,
        error: str,
        response: RerankProviderResponse | None = None,
    ) -> RerankingOutcome:
        return RerankingOutcome(
            memory_pack=memory_pack,
            diagnostics=RerankerDiagnostics(
                status="failed",
                attempted=attempted,
                candidate_count=len(memory_pack.memories),
                model=(
                    response.model
                    if response is not None and response.model
                    else self._settings.model
                ),
                response_id=(
                    response.response_id if response is not None else None
                ),
                usage=(
                    _token_usage(response.total_tokens)
                    if response is not None
                    else None
                ),
                error=error,
                provider=response.provider if response is not None else None,
                search_units=(
                    response.search_units if response is not None else None
                ),
            ),
        )


def _reranker_document(memory: object) -> str:
    return json.dumps(
        {
            "content": str(getattr(memory, "content")),
            "created_at": _timestamp(getattr(memory, "created_at")),
            "valid_from": _optional_timestamp(getattr(memory, "valid_from")),
            "valid_to": _optional_timestamp(getattr(memory, "valid_to")),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_results(
    *,
    results: tuple[RerankProviderResult, ...],
    candidate_count: int,
) -> None:
    indices = tuple(result.index for result in results)
    if len(indices) != candidate_count or set(indices) != set(
        range(candidate_count)
    ):
        raise ValueError(
            "reranker results must contain every candidate exactly once"
        )


def _token_usage(total_tokens: int | None) -> ModelTokenUsage | None:
    if total_tokens is None:
        return None
    return ModelTokenUsage(
        input_tokens=total_tokens,
        output_tokens=0,
        total_tokens=total_tokens,
    )


def _optional_timestamp(value: object) -> str | None:
    return _timestamp(value) if isinstance(value, datetime) else None


def _timestamp(value: object) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("reranker timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _error_text(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {detail}"
