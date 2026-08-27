"""Composable answer generation that consumes FluxMem's public API."""

from fluxmem_infrastructure.answering.agent import (
    AnswerResult,
    AnswerTurnResult,
    AnsweringAgent,
)
from fluxmem_infrastructure.answering.generator import (
    AnswerGenerator,
    AnswerModelSettings,
)
from fluxmem_infrastructure.answering.context import AnswerContextSettings
from fluxmem_infrastructure.answering.reranker import (
    MemoryReranker,
    MemoryRerankerSettings,
    ModelMemoryReranker,
    RerankProvider,
    RerankProviderResponse,
    RerankProviderResult,
    RerankerCandidateDiagnostics,
    RerankerDiagnostics,
    RerankingOutcome,
)

__all__ = (
    "AnswerGenerator",
    "AnswerContextSettings",
    "AnswerModelSettings",
    "AnswerResult",
    "AnswerTurnResult",
    "AnsweringAgent",
    "MemoryReranker",
    "MemoryRerankerSettings",
    "ModelMemoryReranker",
    "RerankProvider",
    "RerankProviderResponse",
    "RerankProviderResult",
    "RerankerCandidateDiagnostics",
    "RerankerDiagnostics",
    "RerankingOutcome",
)
