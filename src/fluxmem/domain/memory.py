from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


def validate_memory_fields(
    *,
    content: str,
    valid_from: datetime | None,
    valid_to: datetime | None,
) -> None:
    """Validate canonical fields shared by stored and proposed memories."""

    if not content.strip():
        raise ValueError("memory content cannot be blank")
    for field, value in (("valid_from", valid_from), ("valid_to", valid_to)):
        if value is not None and value.utcoffset() is None:
            raise ValueError(f"memory {field} must include a timezone")
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ValueError("memory validity cannot end before it begins")


@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: UUID
    message_id: UUID
    content: str
    created_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    session_applicability: UUID | None = None

    def __post_init__(self) -> None:
        if self.created_at.utcoffset() is None:
            raise ValueError("memory created_at must include a timezone")
        validate_memory_fields(
            content=self.content,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
        )
