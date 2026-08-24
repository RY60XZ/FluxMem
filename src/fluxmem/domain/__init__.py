"""Database-independent objects used by FluxMem application code."""

from fluxmem.domain.conflict import (
    ConflictNeighbor,
    ConflictProposal,
    MemoryConflict,
)
from fluxmem.domain.info_pack import (
    FeedbackPack,
    MemoryPack,
    MemoryRetrievalResult,
    MemoryUsage,
    MessagePack,
    RetrievedMemory,
    TurnMemoryPacks,
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
    ReconciliationAction,
    ReconciliationDecision,
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
    "ConflictNeighbor",
    "ConflictProposal",
    "EMBEDDING_DIMENSIONS",
    "DecisionSource",
    "DecayClass",
    "Embedding",
    "IndexStatus",
    "LifecycleDecision",
    "Memory",
    "MemoryLifecycle",
    "MemoryConflict",
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
    "ReconciliationAction",
    "ReconciliationDecision",
    "Message",
    "MessageIngestionResult",
    "MessagePack",
    "RetrievedMemory",
    "TurnMemoryPacks",
    "Session",
    "Status",
    "Tier",
    "UsageType",
    "User",
)
