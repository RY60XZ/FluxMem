from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, TypeVar
from uuid import UUID

from fluxmem.application.lifecycle import tier_for_importance
from fluxmem.application.llm.context import (
    IdReferenceMap,
    LLMContextSettings,
    format_prompt_timestamp,
    render_memory_pack,
    render_messages,
)
from fluxmem.application.llm.prompt_loader import (
    load_prompt,
    render_repair_prompt,
)
from fluxmem.application.ports.lifecycle import (
    LifecycleEvaluationError,
    LifecycleEvaluationInput,
)
from fluxmem.application.ports.llm import (
    InvalidModelOutputError,
    ModelDiagnosticsRecorder,
    ModelProviderError,
    ModelUsageRecorder,
    StructuredModelProvider,
)
from fluxmem.domain.info_pack import MemoryPack, MessagePack
from fluxmem.domain.lifecycle import DecisionSource, LifecycleDecision
from fluxmem.domain.llm import (
    LLMTaskKind,
    ModelCallDiagnostics,
    ModelCallUsage,
    ProposedMemory,
)
from fluxmem.domain.memory import Memory
from fluxmem.domain.message import Message


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class LLMTaskSettings:
    model: str
    timeout_seconds: float = 30.0
    maximum_output_tokens: int = 2_048
    repair_invalid_output: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM task model cannot be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout must be positive")
        if self.maximum_output_tokens < 1:
            raise ValueError("LLM output-token limit must be positive")


class _StructuredTask:
    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        settings: LLMTaskSettings,
    ) -> None:
        self._provider = provider
        self._settings = settings

    def _generate(
        self,
        *,
        task: LLMTaskKind,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any], int], _T],
        usage_recorder: ModelUsageRecorder | None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None,
        supplied_memory_ids: tuple[UUID, ...] = (),
        supplied_message_ids: tuple[UUID, ...] = (),
    ) -> _T:
        try:
            response = self._provider.generate(
                model=self._settings.model,
                instructions=instructions,
                input_text=input_text,
                schema_name=schema_name,
                schema=schema,
                timeout_seconds=self._settings.timeout_seconds,
                maximum_output_tokens=self._settings.maximum_output_tokens,
            )
        except Exception as error:
            _record_unknown_usage(
                recorder=usage_recorder,
                task=task,
                attempt=1,
                model=self._settings.model,
            )
            _record_model_diagnostics(
                recorder=diagnostics_recorder,
                task=task,
                attempt=1,
                requested_model=self._settings.model,
                instructions=instructions,
                input_text=input_text,
                schema_name=schema_name,
                schema=schema,
                response=None,
                validated_output=None,
                error=error,
                supplied_memory_ids=supplied_memory_ids,
                supplied_message_ids=supplied_message_ids,
            )
            raise
        _record_structured_usage(
            recorder=usage_recorder,
            task=task,
            attempt=1,
            response=response,
        )
        try:
            validated = validator(_parse_object(response.output_text), 1)
        except (KeyError, TypeError, ValueError) as first_error:
            _record_model_diagnostics(
                recorder=diagnostics_recorder,
                task=task,
                attempt=1,
                requested_model=self._settings.model,
                instructions=instructions,
                input_text=input_text,
                schema_name=schema_name,
                schema=schema,
                response=response,
                validated_output=None,
                error=first_error,
                supplied_memory_ids=supplied_memory_ids,
                supplied_message_ids=supplied_message_ids,
            )
            if not self._settings.repair_invalid_output:
                raise InvalidModelOutputError(str(first_error)) from first_error
            validation_error = str(first_error)
        else:
            _record_model_diagnostics(
                recorder=diagnostics_recorder,
                task=task,
                attempt=1,
                requested_model=self._settings.model,
                instructions=instructions,
                input_text=input_text,
                schema_name=schema_name,
                schema=schema,
                response=response,
                validated_output=validated,
                error=None,
                supplied_memory_ids=supplied_memory_ids,
                supplied_message_ids=supplied_message_ids,
            )
            return validated

        repair_input = (
            f"{input_text}\n\n"
            f"{render_repair_prompt(validation_error=validation_error)}"
        )
        try:
            repaired = self._provider.generate(
                model=self._settings.model,
                instructions=instructions,
                input_text=repair_input,
                schema_name=schema_name,
                schema=schema,
                timeout_seconds=self._settings.timeout_seconds,
                maximum_output_tokens=self._settings.maximum_output_tokens,
            )
        except Exception as error:
            _record_unknown_usage(
                recorder=usage_recorder,
                task=task,
                attempt=2,
                model=self._settings.model,
            )
            _record_model_diagnostics(
                recorder=diagnostics_recorder,
                task=task,
                attempt=2,
                requested_model=self._settings.model,
                instructions=instructions,
                input_text=repair_input,
                schema_name=schema_name,
                schema=schema,
                response=None,
                validated_output=None,
                error=error,
                supplied_memory_ids=supplied_memory_ids,
                supplied_message_ids=supplied_message_ids,
            )
            raise
        _record_structured_usage(
            recorder=usage_recorder,
            task=task,
            attempt=2,
            response=repaired,
        )
        try:
            validated = validator(_parse_object(repaired.output_text), 2)
        except (KeyError, TypeError, ValueError) as repair_error:
            _record_model_diagnostics(
                recorder=diagnostics_recorder,
                task=task,
                attempt=2,
                requested_model=self._settings.model,
                instructions=instructions,
                input_text=repair_input,
                schema_name=schema_name,
                schema=schema,
                response=repaired,
                validated_output=None,
                error=repair_error,
                supplied_memory_ids=supplied_memory_ids,
                supplied_message_ids=supplied_message_ids,
            )
            raise InvalidModelOutputError(str(repair_error)) from repair_error
        _record_model_diagnostics(
            recorder=diagnostics_recorder,
            task=task,
            attempt=2,
            requested_model=self._settings.model,
            instructions=instructions,
            input_text=repair_input,
            schema_name=schema_name,
            schema=schema,
            response=repaired,
            validated_output=validated,
            error=None,
            supplied_memory_ids=supplied_memory_ids,
            supplied_message_ids=supplied_message_ids,
        )
        return validated


def _parse_object(output_text: str) -> Mapping[str, Any]:
    value = json.loads(output_text)
    if not isinstance(value, dict):
        raise TypeError("structured model output must be an object")
    return value


def _record_structured_usage(
    *,
    recorder: ModelUsageRecorder | None,
    task: LLMTaskKind,
    attempt: int,
    response: object,
) -> None:
    if recorder is None:
        return
    recorder.record(
        ModelCallUsage(
            task=task,
            model=str(getattr(response, "model")),
            attempt=attempt,
            response_id=getattr(response, "response_id", None),
            token_usage=getattr(response, "usage", None),
        )
    )


def _record_unknown_usage(
    *,
    recorder: ModelUsageRecorder | None,
    task: LLMTaskKind,
    attempt: int,
    model: str,
) -> None:
    if recorder is None:
        return
    recorder.record(
        ModelCallUsage(
            task=task,
            model=model,
            attempt=attempt,
            response_id=None,
            token_usage=None,
        )
    )


def _record_model_diagnostics(
    *,
    recorder: ModelDiagnosticsRecorder | None,
    task: LLMTaskKind,
    attempt: int,
    requested_model: str,
    instructions: str,
    input_text: str,
    schema_name: str | None,
    schema: Mapping[str, Any] | None,
    response: object | None,
    validated_output: object | None,
    error: BaseException | None,
    output_text_override: str | None = None,
    supplied_memory_ids: tuple[UUID, ...] = (),
    supplied_message_ids: tuple[UUID, ...] = (),
) -> None:
    if recorder is None:
        return
    response_model = getattr(response, "model", None)
    model = (
        response_model
        if isinstance(response_model, str) and response_model.strip()
        else requested_model
    )
    request_instructions = getattr(response, "request_instructions", None)
    request_input_text = getattr(response, "request_input_text", None)
    detail = _exception_chain_text(error) if error is not None else None
    recorder.record(
        ModelCallDiagnostics(
            task=task,
            model=model,
            attempt=attempt,
            instructions=(
                request_instructions
                if isinstance(request_instructions, str)
                else instructions
            ),
            input_text=(
                request_input_text
                if isinstance(request_input_text, str)
                else input_text
            ),
            schema_name=schema_name,
            schema=dict(schema) if schema is not None else None,
            supplied_memory_ids=supplied_memory_ids,
            supplied_message_ids=supplied_message_ids,
            output_text=(
                output_text_override
                if output_text_override is not None
                else getattr(response, "output_text", None)
            ),
            validated_output=validated_output,
            response_id=getattr(response, "response_id", None),
            token_usage=getattr(response, "usage", None),
            error=detail,
        )
    )


def _exception_chain_text(error: BaseException) -> str:
    """Keep provider context hidden by application-level exception wrappers."""

    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or type(current).__name__
        parts.append(f"{type(current).__name__}: {message}")
        current = current.__cause__ or current.__context__
    return " <- caused by ".join(parts)


def _require_exact_keys(
    value: Mapping[str, Any], *, expected: set[str], object_name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{object_name} fields must be exactly {sorted(expected)}"
        )


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _optional_datetime(value: object, *, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO-8601 string or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


_EXTRACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["memories"],
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "source_message_ref",
                    "session_limited",
                    "valid_from",
                    "valid_to",
                ],
                "properties": {
                    "content": {"type": "string"},
                    "source_message_ref": {"type": "integer"},
                    "session_limited": {"type": "boolean"},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                },
            },
        }
    },
}


class LLMMemoryExtractor(_StructuredTask):
    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        settings: LLMTaskSettings,
        context_settings: LLMContextSettings | None = None,
    ) -> None:
        super().__init__(provider=provider, settings=settings)
        self._context_settings = context_settings or LLMContextSettings()

    def extract(
        self,
        *,
        target_messages: tuple[Message, ...],
        session_history: MessagePack,
        memory_pack: MemoryPack,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[ProposedMemory, ...]:
        if not target_messages:
            return ()
        if any(
            message.session_id != session_history.session_id
            for message in target_messages
        ):
            raise ValueError("extraction targets must share the history session")
        if memory_pack.session_id != session_history.session_id:
            raise ValueError(
                "extraction memory pack and history must share one session"
            )
        target_source_ids = {
            message.message_id for message in target_messages
        }
        context = render_messages(
            message_pack=session_history,
            settings=self._context_settings,
            extra_messages=target_messages,
            maximum_messages=(
                self._context_settings.maximum_write_history_messages
            ),
            maximum_characters=(
                self._context_settings.maximum_write_history_characters
            ),
            maximum_content_characters=(
                self._context_settings.maximum_write_message_content_characters
            ),
        )
        target_source_references = tuple(
            context.message_references.reference_for(message.message_id)
            for message in target_messages
            if context.message_references.contains(message.message_id)
        )
        if len(target_source_references) != len(target_messages):
            raise ValueError("an extraction target is absent from bounded context")
        memories = render_memory_pack(
            memory_pack=memory_pack,
            settings=self._context_settings,
            maximum_characters=(
                self._context_settings.maximum_write_memory_characters
            ),
            include_references=False,
        )

        def validate(
            value: Mapping[str, Any], attempt: int
        ) -> tuple[ProposedMemory, ...]:
            del attempt
            _require_exact_keys(
                value,
                expected={"memories"},
                object_name="extraction result",
            )
            raw_memories = value["memories"]
            if not isinstance(raw_memories, list):
                raise TypeError("memories must be an array")
            if len(raw_memories) > self._context_settings.maximum_memory_candidates:
                raise ValueError("too many memory candidates")

            proposals: list[ProposedMemory] = []
            for raw_memory in raw_memories:
                if not isinstance(raw_memory, dict):
                    raise TypeError("each memory candidate must be an object")
                _require_exact_keys(
                    raw_memory,
                    expected={
                        "content",
                        "source_message_ref",
                        "session_limited",
                        "valid_from",
                        "valid_to",
                    },
                    object_name="memory candidate",
                )
                source_id = context.message_references.id_for(
                    raw_memory["source_message_ref"],
                    field="source_message_ref",
                )
                if source_id not in target_source_ids:
                    raise ValueError(
                        "memory source must identify an extraction target"
                    )
                session_limited = raw_memory["session_limited"]
                if not isinstance(session_limited, bool):
                    raise TypeError("session_limited must be a boolean")
                content = raw_memory["content"]
                if not isinstance(content, str):
                    raise TypeError("memory content must be a string")
                normalized = " ".join(content.split())
                session_applicability = (
                    session_history.session_id if session_limited else None
                )
                valid_from = _optional_datetime(
                    raw_memory["valid_from"], field="valid_from"
                )
                valid_to = _optional_datetime(
                    raw_memory["valid_to"], field="valid_to"
                )
                proposals.append(
                    ProposedMemory(
                        content=normalized,
                        source_message_id=source_id,
                        session_applicability=session_applicability,
                        valid_from=valid_from,
                        valid_to=valid_to,
                    )
                )
            return tuple(proposals)

        return self._generate(
            task=LLMTaskKind.EXTRACTION,
            instructions=load_prompt("memory_extraction"),
            input_text=(
                "TARGET MESSAGE REFS:\n"
                f"{json.dumps(target_source_references)}\n\n"
                f"RECENT CONVERSATION:\n{context.text}"
                f"\n\nMEMORIES:\n{memories.text}"
            ),
            schema_name="fluxmem_memory_extraction",
            schema=_EXTRACTION_SCHEMA,
            validator=validate,
            usage_recorder=usage_recorder,
            diagnostics_recorder=diagnostics_recorder,
            supplied_memory_ids=memories.included_memory_ids,
            supplied_message_ids=context.message_references.ids,
        )


_LIFECYCLE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "memory_ref",
                    "importance",
                    "reason_codes",
                    "confidence",
                ],
                "properties": {
                    "memory_ref": {"type": "integer"},
                    "importance": {"type": "number"},
                    "reason_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}


class LLMLifecycleEvaluator(_StructuredTask):
    def evaluate_many(
        self,
        *,
        items: tuple[LifecycleEvaluationInput, ...],
        evaluated_at: datetime,
        usage_recorder: ModelUsageRecorder | None = None,
        diagnostics_recorder: ModelDiagnosticsRecorder | None = None,
    ) -> tuple[LifecycleDecision | None, ...]:
        if not items:
            return ()
        references = IdReferenceMap(
            tuple(item.memory.memory_id for item in items)
        )
        input_text = json.dumps(
            {
                "memories": [
                    {
                        "memory_ref": reference,
                        "content": item.memory.content,
                        "source_role": item.source_role,
                        "session_limited": (
                            item.memory.session_applicability is not None
                        ),
                        "valid_from": (
                            format_prompt_timestamp(item.memory.valid_from)
                            if item.memory.valid_from is not None
                            else None
                        ),
                        "valid_to": (
                            format_prompt_timestamp(item.memory.valid_to)
                            if item.memory.valid_to is not None
                            else None
                        ),
                    }
                    for reference, item in enumerate(items, start=1)
                ],
                "evaluated_at": format_prompt_timestamp(evaluated_at),
            },
            separators=(",", ":"),
        )

        def validate(
            value: Mapping[str, Any],
            attempt: int,
        ) -> tuple[LifecycleDecision | None, ...]:
            _require_exact_keys(
                value,
                expected={"decisions"},
                object_name="lifecycle batch result",
            )
            raw_decisions = value["decisions"]
            if not isinstance(raw_decisions, list):
                raise TypeError("lifecycle decisions must be an array")
            decisions: dict[int, LifecycleDecision] = {}
            invalid_references: set[int] = set()
            seen_references: set[int] = set()
            for raw in raw_decisions:
                if not isinstance(raw, Mapping):
                    continue
                reference = raw.get("memory_ref")
                try:
                    references.id_for(reference, field="memory_ref")
                except (TypeError, ValueError):
                    continue
                assert isinstance(reference, int)
                if reference in seen_references:
                    invalid_references.add(reference)
                    decisions.pop(reference, None)
                    continue
                seen_references.add(reference)
                try:
                    _require_exact_keys(
                        raw,
                        expected={
                            "memory_ref",
                            "importance",
                            "reason_codes",
                            "confidence",
                        },
                        object_name="lifecycle decision",
                    )
                    importance = _number(
                        raw["importance"], field="importance"
                    )
                    reasons = raw["reason_codes"]
                    if not isinstance(reasons, list) or not all(
                        isinstance(reason, str) and reason.strip()
                        for reason in reasons
                    ):
                        raise TypeError(
                            "reason_codes must be non-empty strings"
                        )
                    decisions[reference] = LifecycleDecision(
                        importance=importance,
                        tier=tier_for_importance(importance),
                        initial_retention=importance,
                        reason_codes=tuple(
                            reason.strip() for reason in reasons
                        ),
                        confidence=_number(
                            raw["confidence"], field="confidence"
                        ),
                        decision_source=(
                            DecisionSource.LLM_PRIMARY
                            if attempt == 1
                            else DecisionSource.LLM_RETRY
                        ),
                    )
                except (KeyError, TypeError, ValueError):
                    invalid_references.add(reference)
            return tuple(
                decisions.get(reference)
                if reference not in invalid_references
                else None
                for reference in range(1, len(items) + 1)
            )

        try:
            return self._generate(
                task=LLMTaskKind.LIFECYCLE,
                instructions=load_prompt("lifecycle_evaluation"),
                input_text=input_text,
                schema_name="fluxmem_lifecycle_decisions",
                schema=_LIFECYCLE_SCHEMA,
                validator=validate,
                usage_recorder=usage_recorder,
                diagnostics_recorder=diagnostics_recorder,
                supplied_memory_ids=references.ids,
            )
        except ModelProviderError as error:
            raise LifecycleEvaluationError(
                "lifecycle model could not produce a usable decision"
            ) from error
