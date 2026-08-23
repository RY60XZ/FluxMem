from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from typing import Protocol
from uuid import UUID

from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.message import Message


@dataclass(frozen=True, slots=True)
class LLMContextSettings:
    maximum_history_messages: int = 128
    maximum_answer_input_tokens: int = 64_000
    maximum_write_history_messages: int = 6
    maximum_write_history_characters: int = 4_000
    maximum_write_message_content_characters: int = 1_000
    maximum_memory_characters: int = 16_000
    maximum_write_memory_characters: int = 8_000
    maximum_memory_content_characters: int = 2_000
    maximum_memory_candidates: int = 10

    def __post_init__(self) -> None:
        if self.maximum_history_messages < 1:
            raise ValueError("history message limit must be positive")
        if self.maximum_answer_input_tokens < 1:
            raise ValueError("answer input-token budget must be positive")
        if self.maximum_write_history_messages < 1:
            raise ValueError("write-history message limit must be positive")
        if self.maximum_write_history_characters < 1:
            raise ValueError("write-history character limit must be positive")
        if self.maximum_write_message_content_characters < 1:
            raise ValueError("write-history content limit must be positive")
        if self.maximum_memory_characters < 1:
            raise ValueError("memory character limit must be positive")
        if self.maximum_write_memory_characters < 1:
            raise ValueError("write-memory character limit must be positive")
        if self.maximum_memory_content_characters < 1:
            raise ValueError("memory content limit must be positive")
        if self.maximum_memory_candidates < 1:
            raise ValueError("memory candidate limit must be positive")


@dataclass(frozen=True, slots=True)
class IdReferenceMap:
    """A prompt-local, 1-based reference map for canonical UUIDs."""

    ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("reference-map IDs must be unique")

    def reference_for(self, canonical_id: UUID) -> int:
        try:
            return self.ids.index(canonical_id) + 1
        except ValueError as error:
            raise ValueError("canonical ID is absent from prompt references") from error

    def id_for(self, reference: object, *, field: str) -> UUID:
        if isinstance(reference, bool) or not isinstance(reference, int):
            raise TypeError(f"{field} must be an integer reference")
        if reference < 1 or reference > len(self.ids):
            raise ValueError(f"{field} is absent from prompt references")
        return self.ids[reference - 1]

    def contains(self, canonical_id: UUID) -> bool:
        return canonical_id in self.ids


@dataclass(frozen=True, slots=True)
class RenderedMessageContext:
    text: str
    message_references: IdReferenceMap


@dataclass(frozen=True, slots=True)
class RenderedMemoryContext:
    text: str
    memory_references: IdReferenceMap

    @property
    def included_memory_ids(self) -> tuple[UUID, ...]:
        return self.memory_references.ids


class TokenCounter(Protocol):
    """Estimate preflight tokens before provider-native usage is available."""

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ApproximateTokenCounter:
    """Portable UTF-8 estimate; actual provider usage remains authoritative."""

    utf8_bytes_per_token: float = 3.0

    def __post_init__(self) -> None:
        if self.utf8_bytes_per_token <= 0:
            raise ValueError("UTF-8 bytes per token must be positive")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return ceil(len(text.encode("utf-8")) / self.utf8_bytes_per_token)


def _clip(value: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _bounded_json_line(
    *, record: dict[str, object], text_field: str, limit: int
) -> str | None:
    def encode(content: str) -> str:
        bounded = dict(record)
        bounded[text_field] = content
        return json.dumps(
            bounded,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    content = str(record[text_field])
    rendered = encode(content)
    if len(rendered) <= limit:
        return rendered
    if len(encode("")) > limit:
        return None

    low = 0
    high = len(content)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = _clip(content, limit=middle)
        encoded = encode(candidate)
        if len(encoded) <= limit:
            best = encoded
            low = middle + 1
        else:
            high = middle - 1
    return best


def _bounded_json_line_by_tokens(
    *,
    record: dict[str, object],
    text_field: str,
    limit: int,
    token_counter: TokenCounter,
) -> str | None:
    def encode(content: str) -> str:
        bounded = dict(record)
        bounded[text_field] = content
        return json.dumps(
            bounded,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    content = str(record[text_field])
    rendered = encode(content)
    if token_counter.count(rendered) <= limit:
        return rendered
    if token_counter.count(encode("")) > limit:
        return None

    low = 0
    high = len(content)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = _clip(content, limit=middle)
        encoded = encode(candidate)
        if token_counter.count(encoded) <= limit:
            best = encoded
            low = middle + 1
        else:
            high = middle - 1
    return best


def render_messages(
    *,
    message_pack: MessagePack,
    settings: LLMContextSettings,
    extra_messages: tuple[Message, ...] = (),
    maximum_messages: int | None = None,
    maximum_tokens: int | None = None,
    maximum_characters: int | None = None,
    maximum_content_characters: int | None = None,
    token_counter: TokenCounter | None = None,
) -> RenderedMessageContext:
    """Render recent messages with prompt-local integer references."""

    message_limit = (
        settings.maximum_history_messages
        if maximum_messages is None
        else maximum_messages
    )
    if maximum_tokens is not None and maximum_characters is not None:
        raise ValueError("message context accepts one size budget")
    use_character_budget = maximum_characters is not None
    size_limit = (
        maximum_characters
        if use_character_budget
        else (
            settings.maximum_answer_input_tokens
            if maximum_tokens is None
            else maximum_tokens
        )
    )
    if message_limit < 1 or size_limit < 1:
        raise ValueError("message context limits must be positive")
    if (
        maximum_content_characters is not None
        and maximum_content_characters < 1
    ):
        raise ValueError("message content limit must be positive")

    seen: set[UUID] = set()
    combined: list[Message] = []
    for message in (*message_pack.messages, *extra_messages):
        if message.message_id in seen:
            continue
        if message.session_id != message_pack.session_id:
            raise ValueError("LLM context messages must share one session")
        seen.add(message.message_id)
        combined.append(message)

    selected = combined[-message_limit:]
    rendered_reversed: list[tuple[Message, dict[str, object]]] = []
    counter = token_counter or ApproximateTokenCounter()
    remaining = size_limit
    for message in reversed(selected):
        raw_record = {
            "message_ref": message_limit,
            "role": message.role,
            "content": (
                _clip(message.content, limit=maximum_content_characters)
                if maximum_content_characters is not None
                else message.content
            ),
            "created_at": message.created_at.isoformat(),
        }
        if use_character_budget:
            record = _bounded_json_line(
                record=raw_record,
                text_field="content",
                limit=remaining,
            )
        else:
            record = _bounded_json_line_by_tokens(
                record=raw_record,
                text_field="content",
                limit=remaining,
                token_counter=counter,
            )
        if record is None:
            break
        rendered_reversed.append((message, json.loads(record)))
        remaining -= (
            len(record) + 1
            if use_character_budget
            else counter.count(f"{record}\n")
        )
        if remaining <= 0:
            break

    rendered = list(reversed(rendered_reversed))
    lines: list[str] = []
    message_ids: list[UUID] = []
    for reference, (message, record) in enumerate(rendered, start=1):
        record["message_ref"] = reference
        lines.append(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )
        message_ids.append(message.message_id)
    return RenderedMessageContext(
        text="\n".join(lines),
        message_references=IdReferenceMap(tuple(message_ids)),
    )


def render_memory_pack(
    *,
    memory_pack: MemoryPack,
    settings: LLMContextSettings,
    maximum_characters: int | None = None,
    include_references: bool = True,
) -> RenderedMemoryContext:
    """Render bounded returned memories as uniform semantic evidence."""

    character_limit = (
        settings.maximum_memory_characters
        if maximum_characters is None
        else maximum_characters
    )
    if character_limit < 1:
        raise ValueError("memory context limit must be positive")

    lines: list[str] = []
    included_ids: list[UUID] = []
    remaining = character_limit
    for retrieved in memory_pack.memories:
        memory = retrieved.memory
        semantic_record: dict[str, object] = {
            "content": _clip(
                memory.content,
                limit=settings.maximum_memory_content_characters,
            ),
            "valid_from": (
                memory.valid_from.isoformat()
                if memory.valid_from is not None
                else None
            ),
            "valid_to": (
                memory.valid_to.isoformat()
                if memory.valid_to is not None
                else None
            ),
            "created_at": memory.created_at.isoformat(),
        }
        if include_references:
            semantic_record = {
                "memory_ref": len(included_ids) + 1,
                **semantic_record,
            }
        record = _bounded_json_line(
            record=semantic_record,
            text_field="content",
            limit=remaining,
        )
        if record is None:
            break
        lines.append(record)
        included_ids.append(memory.memory_id)
        remaining -= len(record) + 1

    return RenderedMemoryContext(
        text="\n".join(lines),
        memory_references=IdReferenceMap(tuple(included_ids)),
    )
