"""Public API for FluxMem's independent memory layer."""

from fluxmem.api import FluxMem
from fluxmem.application.llm import (
    LLMContextSettings,
    LLMLifecycleEvaluator,
    LLMMemoryExtractor,
    LLMMemoryReconciler,
    LLMTaskSettings,
)
from fluxmem.application.ports.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
)
from fluxmem.application.ports.lifecycle import (
    LifecycleEvaluationInput,
    LifecycleEvaluator,
)
from fluxmem.application.ports.llm import (
    InvalidModelOutputError,
    MemoryExtractor,
    MemoryReconciler,
    ModelProviderError,
    ModelTimeoutError,
    StructuredModelProvider,
    StructuredModelResponse,
)
from fluxmem.application.read import HybridRetrievalSettings
from fluxmem.bootstrap import MemoryLayerSettings, bootstrap
from fluxmem.diagnostics import diagnostics_to_dict
from fluxmem.domain.info_pack import (
    MemoryPack,
    MemoryRetrievalResult,
    MessagePack,
    RetrievedMemory,
)
from fluxmem.domain.llm import (
    LLMUsageReport,
    MemoryDiagnostics,
    MemoryWriteOutcome,
    MemoryWriteStatus,
    MessageIngestionResult,
    ModelTokenUsage,
)
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message, Session, User
from fluxmem.domain.retrieval import EMBEDDING_DIMENSIONS, Embedding

__version__ = "0.1.0"

__all__ = (
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EMBEDDING_DIMENSIONS",
    "Embedding",
    "FluxMem",
    "HybridRetrievalSettings",
    "LLMContextSettings",
    "LLMLifecycleEvaluator",
    "LLMMemoryExtractor",
    "LLMMemoryReconciler",
    "LLMTaskSettings",
    "LLMUsageReport",
    "InvalidModelOutputError",
    "LifecycleEvaluator",
    "LifecycleEvaluationInput",
    "Memory",
    "MemoryDiagnostics",
    "MemoryExtractor",
    "MemoryLayerSettings",
    "MemoryPack",
    "MemoryReconciler",
    "MemoryRetrievalResult",
    "MemoryWriteOutcome",
    "MemoryWriteStatus",
    "Message",
    "MessageIngestionResult",
    "MessagePack",
    "ModelProviderError",
    "ModelTimeoutError",
    "ModelTokenUsage",
    "RetrievedMemory",
    "Session",
    "StructuredModelProvider",
    "StructuredModelResponse",
    "User",
    "__version__",
    "bootstrap",
    "diagnostics_to_dict",
)
