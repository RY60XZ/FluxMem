"""SQLAlchemy mappings imported together for Alembic metadata discovery."""

from fluxmem.adapters.postgres.models.base import Base
from fluxmem.adapters.postgres.models.lifecycle import MemoryLifecycleRow
from fluxmem.adapters.postgres.models.message import MessageRow
from fluxmem.adapters.postgres.models.memory import MemoryIndexRow, MemoryRow
from fluxmem.adapters.postgres.models.memory_usage import MemoryUsageRow
from fluxmem.adapters.postgres.models.retrieval import (
    RetrievalCandidateRow,
    RetrievalQueryRow,
)
from fluxmem.adapters.postgres.models.session import SessionRow, UserRow

__all__ = (
    "Base",
    "MemoryLifecycleRow",
    "MemoryIndexRow",
    "MemoryRow",
    "MemoryUsageRow",
    "MessageRow",
    "RetrievalCandidateRow",
    "RetrievalQueryRow",
    "SessionRow",
    "UserRow",
)
