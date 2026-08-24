from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from uuid import UUID

from fluxmem import MemoryPack, Message, MessagePack, ModelTokenUsage
from fluxmem_infrastructure.answering.context import (
    AnswerContextSettings,
    render_answer_context,
)
from fluxmem_infrastructure.answering.ports import TextStreamingModelProvider


@dataclass(frozen=True, slots=True)
class AnswerModelSettings:
    model: str
    timeout_seconds: float = 60.0
    maximum_output_tokens: int = 2_048

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("answer model cannot be blank")
        if self.timeout_seconds <= 0 or self.maximum_output_tokens < 1:
            raise ValueError("answer model limits must be positive")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    content: str
    context_memory_ids: tuple[UUID, ...]
    model: str
    response_id: str | None
    usage: ModelTokenUsage | None


class AnswerGenerator:
    def __init__(
        self,
        *,
        provider: TextStreamingModelProvider,
        settings: AnswerModelSettings,
        context_settings: AnswerContextSettings | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._context_settings = context_settings or AnswerContextSettings()

    @property
    def model(self) -> str:
        return self._settings.model

    def generate(
        self,
        *,
        query: Message,
        history: MessagePack,
        memory_pack: MemoryPack,
    ) -> GeneratedAnswer:
        instructions = _answer_prompt()
        context = render_answer_context(
            query=query,
            history=history,
            memories=memory_pack,
            instructions=instructions,
            settings=self._context_settings,
        )
        stream = self._provider.stream_text(
            model=self._settings.model,
            instructions=instructions,
            input_text=context.input_text,
            timeout_seconds=self._settings.timeout_seconds,
            maximum_output_tokens=self._settings.maximum_output_tokens,
            input_text_blocks=context.input_blocks,
            prompt_cache_key=_cache_key(
                user_id=memory_pack.user_id,
                session_id=memory_pack.session_id,
                instructions=instructions,
            ),
        )
        try:
            content = "".join(stream).strip()
        finally:
            stream.close()
        if not content:
            raise ValueError("generated answer cannot be blank")
        model = stream.model if stream.model.strip() else self._settings.model
        return GeneratedAnswer(
            content=content,
            context_memory_ids=context.memory_ids,
            model=model,
            response_id=stream.response_id,
            usage=stream.usage,
        )


def _cache_key(*, user_id: UUID, session_id: UUID, instructions: str) -> str:
    digest = hashlib.sha256()
    for value in ("answering-agent-v1", str(user_id), str(session_id), instructions):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return f"answering-agent:{digest.hexdigest()[:32]}"


def _answer_prompt() -> str:
    return (
        files("fluxmem_infrastructure.answering.prompts")
        .joinpath("answer.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
