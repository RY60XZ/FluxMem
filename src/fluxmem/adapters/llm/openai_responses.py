from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any, Mapping

from fluxmem.application.ports.llm import (
    ModelInputTextBlock,
    ModelProviderError,
    ModelTimeoutError,
    StructuredModelResponse,
)
from fluxmem.domain.llm import ModelTokenUsage


class _OpenAITextStream(Iterator[str]):
    """Lazy OpenAI stream with terminal response metadata."""

    def __init__(self, *, chunks: Iterable[str], model: str) -> None:
        self._chunks = iter(chunks)
        self.model = model
        self.response_id: str | None = None
        self.usage: ModelTokenUsage | None = None

    def __iter__(self) -> _OpenAITextStream:
        return self

    def __next__(self) -> str:
        return next(self._chunks)

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    def record_completion(self, response: object) -> None:
        model = _field(response, "model")
        if isinstance(model, str) and model.strip():
            self.model = model
        response_id = _field(response, "id")
        if response_id:
            self.response_id = str(response_id)
        self.usage = _model_token_usage(response)


class OpenAIResponsesProvider:
    """Structured generation and text streaming through OpenAI Responses."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        explicit_prompt_caching: bool | None = None,
    ) -> None:
        self._explicit_prompt_caching = explicit_prompt_caching
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI adapter requires the optional 'llm-openai' dependency"
            ) from error
        self._client = OpenAI(api_key=api_key)

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
    ) -> StructuredModelResponse:
        try:
            request_client = self._client
            if hasattr(request_client, "with_options"):
                request_client = request_client.with_options(
                    timeout=timeout_seconds
                )
            response = request_client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": dict(schema),
                        "strict": True,
                    }
                },
                max_output_tokens=maximum_output_tokens,
                store=False,
            )
        except TimeoutError as error:
            raise ModelTimeoutError("OpenAI response timed out") from error
        except Exception as error:
            if "timeout" in type(error).__name__.lower():
                raise ModelTimeoutError("OpenAI response timed out") from error
            raise ModelProviderError("OpenAI response request failed") from error

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ModelProviderError("OpenAI response contained no output text")
        return StructuredModelResponse(
            output_text=output_text,
            model=str(getattr(response, "model", model)),
            response_id=(
                str(response.id) if getattr(response, "id", None) else None
            ),
            usage=_model_token_usage(response),
            request_instructions=instructions,
            request_input_text=input_text,
        )

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
    ) -> _OpenAITextStream:
        """Yield deltas and expose usage after the terminal event arrives."""

        response_input, cache_options = _answer_response_input(
            model=model,
            input_text=input_text,
            input_text_blocks=input_text_blocks,
            prompt_cache_key=prompt_cache_key,
            explicit_prompt_caching=self._explicit_prompt_caching,
        )
        result: _OpenAITextStream

        def chunks() -> Iterator[str]:
            stream = None
            received_text = False
            try:
                request_client = self._client
                if hasattr(request_client, "with_options"):
                    request_client = request_client.with_options(
                        timeout=timeout_seconds
                    )
                request: dict[str, object] = {
                    "model": model,
                    "instructions": instructions,
                    "input": response_input,
                    "max_output_tokens": maximum_output_tokens,
                    "store": False,
                    "stream": True,
                }
                if cache_options:
                    # extra_body keeps compatibility with older SDK clients
                    # while sending current Responses API cache fields.
                    request["extra_body"] = cache_options
                stream = request_client.responses.create(**request)
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, str) and delta:
                            received_text = True
                            yield delta
                    elif event_type in {
                        "response.completed",
                        "response.incomplete",
                    }:
                        completed_response = getattr(event, "response", None)
                        if completed_response is not None:
                            result.record_completion(completed_response)
                    elif event_type in {"error", "response.failed"}:
                        failed_response = getattr(event, "response", None)
                        if failed_response is not None:
                            result.record_completion(failed_response)
                        raise ModelProviderError(_stream_error_message(event))
            except ModelProviderError:
                raise
            except TimeoutError as error:
                raise ModelTimeoutError("OpenAI response timed out") from error
            except Exception as error:
                if "timeout" in type(error).__name__.lower():
                    raise ModelTimeoutError("OpenAI response timed out") from error
                raise ModelProviderError(
                    "OpenAI response request failed"
                ) from error
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

            if not received_text:
                raise ModelProviderError(
                    "OpenAI response contained no output text"
                )

        result = _OpenAITextStream(chunks=chunks(), model=model)
        return result


def _answer_response_input(
    *,
    model: str,
    input_text: str,
    input_text_blocks: tuple[ModelInputTextBlock, ...],
    prompt_cache_key: str | None,
    explicit_prompt_caching: bool | None,
) -> tuple[object, dict[str, object]]:
    if prompt_cache_key is not None and not prompt_cache_key.strip():
        raise ValueError("prompt cache key cannot be blank")
    if not input_text_blocks:
        return input_text, (
            {"prompt_cache_key": prompt_cache_key}
            if prompt_cache_key is not None
            else {}
        )
    if "".join(block.text for block in input_text_blocks) != input_text:
        raise ValueError("model input blocks must reproduce input_text exactly")

    use_explicit = (
        _supports_explicit_prompt_caching(model)
        if explicit_prompt_caching is None
        else explicit_prompt_caching
    )
    content: list[dict[str, object]] = []
    has_explicit_breakpoint = False
    for block in input_text_blocks:
        rendered: dict[str, object] = {
            "type": "input_text",
            "text": block.text,
        }
        if block.cache_breakpoint and use_explicit:
            rendered["prompt_cache_breakpoint"] = {"mode": "explicit"}
            has_explicit_breakpoint = True
        content.append(rendered)

    cache_options: dict[str, object] = {}
    if prompt_cache_key is not None:
        cache_options["prompt_cache_key"] = prompt_cache_key
    if has_explicit_breakpoint:
        cache_options["prompt_cache_options"] = {"mode": "explicit"}
    return (
        [
            {
                "type": "message",
                "role": "user",
                "content": content,
            }
        ],
        cache_options,
    )


def _supports_explicit_prompt_caching(model: str) -> bool:
    """Use current explicit cache fields only for GPT-5.6+ model names."""

    match = re.search(r"gpt-(\d+)(?:\.(\d+))?", model.casefold())
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (5, 6)


def _stream_error_message(event: object) -> str:
    error = getattr(event, "error", None)
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = "OpenAI streaming response failed"
    return message


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _token_count(value: object, name: str) -> int | None:
    count = _field(value, name)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def _model_token_usage(response: object) -> ModelTokenUsage | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    input_tokens = _token_count(usage, "input_tokens")
    output_tokens = _token_count(usage, "output_tokens")
    total_tokens = _token_count(usage, "total_tokens")
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None

    input_details = _field(usage, "input_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    try:
        return ModelTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=(
                _token_count(input_details, "cached_tokens") or 0
            ),
            cache_write_input_tokens=(
                _token_count(input_details, "cache_write_tokens") or 0
            ),
            reasoning_output_tokens=(
                _token_count(output_details, "reasoning_tokens") or 0
            ),
        )
    except (TypeError, ValueError):
        # Telemetry must not invalidate otherwise usable model output.
        return None
