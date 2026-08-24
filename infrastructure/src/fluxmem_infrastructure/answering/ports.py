from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from fluxmem import ModelTokenUsage


@dataclass(frozen=True, slots=True)
class ModelInputTextBlock:
    text: str
    cache_breakpoint: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("model input text blocks cannot be empty")


class StreamingModelResponse(Iterable[str], Protocol):
    @property
    def model(self) -> str: ...

    @property
    def response_id(self) -> str | None: ...

    @property
    def usage(self) -> ModelTokenUsage | None: ...

    def close(self) -> None: ...


class TextStreamingModelProvider(Protocol):
    def stream_text(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        timeout_seconds: float,
        maximum_output_tokens: int,
        input_text_blocks: tuple[ModelInputTextBlock, ...] = (),
        prompt_cache_key: str | None = None,
    ) -> StreamingModelResponse: ...
