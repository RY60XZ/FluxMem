"""Database-independent objects used by FluxMem application code."""

from fluxmem.domain.info_pack import (
    FeedbackPack,
    MemoryPack,
    MemoryRetrievalResult,
    MemoryUsage,
    MessagePack,
    RetrievedMemory,
    UsageType,
)
from fluxmem.domain.llm import (
    LLMTaskKind,
    LLMUsageReport,
    MemoryDiagnostics,
    MessageIngestionResult,
    MemoryWriteOutcome,
    MemoryWriteStatus,
    ModelCallDiagnostics,
    ModelCallUsage,
    ModelTokenUsage,
    ProposedMemory,
)
from fluxmem.domain.lifecycle import (
    DecisionSource,
    DecayClass,
    LifecycleDecision,
    MemoryLifecycle,
    Status,
    Tier,
)
from fluxmem.domain.memory import (
    Memory,
)
from fluxmem.domain.message import (
    Message,
    Session,
    User,
)
from fluxmem.domain.retrieval import (
    EMBEDDING_DIMENSIONS,
    Embedding,
    IndexStatus,
    MemoryIndex,
    MemorySearchQuery,
)

__all__ = (
    "FeedbackPack",
    "EMBEDDING_DIMENSIONS",
    "DecisionSource",
    "DecayClass",
    "Embedding",
    "IndexStatus",
    "LifecycleDecision",
    "Memory",
    "MemoryLifecycle",
    "MemoryIndex",
    "MemoryPack",
    "MemorySearchQuery",
    "MemoryUsage",
    "LLMTaskKind",
    "LLMUsageReport",
    "MemoryDiagnostics",
    "MemoryRetrievalResult",
    "MemoryWriteOutcome",
    "MemoryWriteStatus",
    "ModelCallDiagnostics",
    "ModelCallUsage",
    "ModelTokenUsage",
    "ProposedMemory",
    "Message",
    "MessageIngestionResult",
    "MessagePack",
    "RetrievedMemory",
    "Session",
    "Status",
    "Tier",
    "UsageType",
    "User",
)
