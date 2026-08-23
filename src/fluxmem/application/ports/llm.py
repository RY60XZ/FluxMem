from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.llm import (
    GeneratedAnswer,
    GeneratedAnswerStream,
    ModelCallDiagnostics,
    ModelCallUsage,
    ModelTokenUsage,
    ProposedMemory,
    ReconciliationDecision,
)
from fluxmem.domain.message import Message


class ModelProviderError(RuntimeError):
    """A structured model request could not produce a usable response."""


class ModelTimeoutError(ModelProviderError):
    """A structured model request exceeded its configured deadline."""


class InvalidModelOutputError(ModelProviderError):
    """A model response failed parsing or application validation."""


@dataclass(frozen=True, slots=True)
class StructuredModelResponse:
    output_text: str
    model: str
    response_id: str | None = None
    usage: ModelTokenUsage | None = None
    request_instructions: str | None = None
    request_input_text: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInputTextBlock:
    """One visible input-text block and its optional cache breakpoint."""

    text: str
    cache_breakpoint: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("model input text blocks cannot be empty")


class StreamingModelResponse(Iterable[str], Protocol):
    """Text deltas whose metadata is populated when the stream completes."""

    @property
    def model(self) -> str: ...

    @property
    def response_id(self) -> str | None: ...

    @property
    def usage(self) -> ModelTokenUsage | None: ...

    def close(self) -> None: ...


class ModelUsageRecorder(Protocol):
    def record(self, call: ModelCallUsage) -> None: ...


class ModelDiagnosticsRecorder(Protocol):
    def record(self, call: ModelCallDiagnostics) -> None: ...


class StructuredModelProvider(Protocol):
    def generate(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        timeout_seconds: float,
        maximum_output_tokens: int,
    ) -> StructuredModelResponse: ...


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


class ModelProvider(
    StructuredModelProvider,
    TextStreamingModelProvider,
    Protocol,
):
    """Model provider used by the complete LLM integration."""


class AnswerGenerator(Protocol):
    def stream(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        memory_pack: MemoryPack,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> GeneratedAnswerStream: ...

    def generate(
        self,
        *,
        message: Message,
        session_history: MessagePack,
        memory_pack: MemoryPack,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> GeneratedAnswer: ...


class MemoryExtractor(Protocol):
    def extract(
        self,
        *,
        target_messages: tuple[Message, ...],
        session_history: MessagePack,
        memory_pack: MemoryPack,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[ProposedMemory, ...]: ...


class MemoryReconciler(Protocol):
    def reconcile(
        self,
        *,
        candidates: tuple[ProposedMemory, ...],
        memory_pack: MemoryPack,
        evidence_messages: tuple[Message, ...],
        session_history: MessagePack,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[ReconciliationDecision, ...]: ...
