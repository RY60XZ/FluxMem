"""Database-independent objects used by FluxMem application code."""

from fluxmem.domain.conflict import (
    ConflictNeighbor,
    ConflictProposal,
    MemoryConflict,
)
from fluxmem.domain.info_pack import (
    FeedbackPack,
    MemoryPack,
    MemoryUsage,
    MessagePack,
    RetrievedMemory,
    TurnMemoryPacks,
    UsageType,
)
from fluxmem.domain.llm import (
    ConversationTurnResult,
    GeneratedAnswer,
    GeneratedAnswerStream,
    LLMTaskKind,
    LLMUsageReport,
    MemoryWriteOutcome,
    MemoryWriteStatus,
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
    QueryType,
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
    "ConversationTurnResult",
    "GeneratedAnswer",
    "GeneratedAnswerStream",
    "LLMTaskKind",
    "LLMUsageReport",
    "MemoryWriteOutcome",
    "MemoryWriteStatus",
    "ModelCallUsage",
    "ModelTokenUsage",
    "ProposedMemory",
    "ReconciliationAction",
    "ReconciliationDecision",
    "Message",
    "MessagePack",
    "RetrievedMemory",
    "TurnMemoryPacks",
    "QueryType",
    "Session",
    "Status",
    "Tier",
    "UsageType",
    "User",
)
