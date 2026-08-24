"""Public API for configuring and running FluxMem."""

from fluxmem.application.process_turn import ConversationTurnStream
from fluxmem.bootstrap import FluxMemServices, bootstrap
from fluxmem.diagnostics import diagnostics_to_dict
from fluxmem.domain.llm import (
    ConversationTurnDiagnostics,
    ConversationTurnResult,
    LLMTaskKind,
    LLMUsageReport,
    MemoryWriteOutcome,
    MemoryWriteStatus,
)
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message, Session, User
from fluxmem.local import (
    LocalRuntimeSettings,
    bootstrap_from_env,
    load_local_settings,
)

__version__ = "0.1.0"

__all__ = (
    "ConversationTurnDiagnostics",
    "ConversationTurnResult",
    "ConversationTurnStream",
    "FluxMemServices",
    "LLMTaskKind",
    "LLMUsageReport",
    "LocalRuntimeSettings",
    "Memory",
    "MemoryWriteOutcome",
    "MemoryWriteStatus",
    "Message",
    "Session",
    "User",
    "__version__",
    "bootstrap",
    "bootstrap_from_env",
    "diagnostics_to_dict",
    "load_local_settings",
)
