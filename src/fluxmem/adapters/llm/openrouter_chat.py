from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any, Mapping

from fluxmem.adapters.openrouter import OPENROUTER_BASE_URL
from fluxmem.application.ports.llm import (
    ModelInputTextBlock,
    ModelProviderError,
    ModelTimeoutError,
    StructuredModelResponse,
)
from fluxmem.domain.llm import ModelTokenUsage


class _OpenRouterTextStream(Iterator[str]):
    """Lazy OpenRouter Chat Completions stream with terminal metadata."""

    def __init__(self, *, chunks: Iterable[str], model: str) -> None:
        self._chunks = iter(chunks)
        self.model = model
        self.response_id: str | None = None
        self.usage: ModelTokenUsage | None = None

    def __iter__(self) -> _OpenRouterTextStream:
        return self

    def __next__(self) -> str:
        return next(self._chunks)

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    def record_chunk(self, chunk: object) -> None:
        model = _field(chunk, "model")
        if isinstance(model, str) and model.strip():
            self.model = model
        response_id = _field(chunk, "id")
        if response_id:
            self.response_id = str(response_id)
        usage = _chat_token_usage(chunk)
        if usage is not None:
            self.usage = usage


class OpenRouterChatCompletionsProvider:
    """FluxMem model provider backed by OpenRouter Chat Completions."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        http_referer: str | None = None,
        app_name: str | None = None,
        strict_json_schema: bool = False,
        enable_prompt_caching: bool = True,
    ) -> None:
        self._strict_json_schema = strict_json_schema
        self._enable_prompt_caching = enable_prompt_caching
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenRouter adapter requires the optional 'local' dependency"
            ) from error

        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if resolved_api_key is None or not resolved_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY cannot be blank")
        if not base_url.strip():
            raise ValueError("OpenRouter base URL cannot be blank")

        default_headers: dict[str, str] = {}
        if http_referer is not None and http_referer.strip():
            default_headers["HTTP-Referer"] = http_referer.strip()
        if app_name is not None and app_name.strip():
            default_headers["X-OpenRouter-Title"] = app_name.strip()
        client_options: dict[str, object] = {
            "api_key": resolved_api_key,
            "base_url": base_url.rstrip("/"),
        }
        if default_headers:
            client_options["default_headers"] = default_headers
        self._client = OpenAI(**client_options)

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
        response_format: dict[str, object]
        system_instructions = instructions
        if self._strict_json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        else:
            # Models such as Gemma 4 expose JSON-object mode but do not enforce
            # JSON Schema. The application still validates and repairs output.
            response_format = {"type": "json_object"}
            system_instructions = _instructions_with_schema(
                instructions=instructions,
                schema=schema,
            )

        try:
            request_client = self._request_client(timeout_seconds)
            response = request_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": input_text},
                ],
                response_format=response_format,
                max_tokens=maximum_output_tokens,
                temperature=0,
                extra_body={"provider": {"require_parameters": True}},
            )
        except TimeoutError as error:
            raise ModelTimeoutError("OpenRouter response timed out") from error
        except Exception as error:
            if "timeout" in type(error).__name__.lower():
                raise ModelTimeoutError("OpenRouter response timed out") from error
            raise ModelProviderError("OpenRouter response request failed") from error

        output_text = _completion_message_text(response)
        if not output_text.strip():
            raise ModelProviderError("OpenRouter response contained no output text")
        return StructuredModelResponse(
            output_text=output_text,
            model=_nonblank_string(_field(response, "model"), fallback=model),
            response_id=(
                str(_field(response, "id"))
                if _field(response, "id")
                else None
            ),
            usage=_chat_token_usage(response),
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
    ) -> _OpenRouterTextStream:
        """Yield answer deltas and retain usage from OpenRouter's final chunk."""

        user_content = _chat_user_content(
            input_text=input_text,
            input_text_blocks=input_text_blocks,
            enable_prompt_caching=self._enable_prompt_caching,
        )
        if prompt_cache_key is not None and not prompt_cache_key.strip():
            raise ValueError("prompt cache key cannot be blank")
        result: _OpenRouterTextStream

        def chunks() -> Iterator[str]:
            stream = None
            received_text = False
            try:
                request_client = self._request_client(timeout_seconds)
                request: dict[str, object] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": maximum_output_tokens,
                    "stream": True,
                }
                if prompt_cache_key is not None:
                    # OpenRouter uses session_id as a sticky routing key, which
                    # improves the chance of provider-side prompt-cache hits.
                    request["extra_body"] = {
                        "session_id": prompt_cache_key,
                    }
                stream = request_client.chat.completions.create(**request)
                for chunk in stream:
                    result.record_chunk(chunk)
                    delta = _completion_delta_text(chunk)
                    if delta:
                        received_text = True
                        yield delta
            except ModelProviderError:
                raise
            except TimeoutError as error:
                raise ModelTimeoutError("OpenRouter response timed out") from error
            except Exception as error:
                if "timeout" in type(error).__name__.lower():
                    raise ModelTimeoutError(
                        "OpenRouter response timed out"
                    ) from error
                raise ModelProviderError(
                    "OpenRouter response request failed"
                ) from error
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

            if not received_text:
                raise ModelProviderError(
                    "OpenRouter response contained no output text"
                )

        result = _OpenRouterTextStream(chunks=chunks(), model=model)
        return result

    def _request_client(self, timeout_seconds: float) -> Any:
        if hasattr(self._client, "with_options"):
            return self._client.with_options(timeout=timeout_seconds)
        return self._client


def _instructions_with_schema(
    *, instructions: str, schema: Mapping[str, Any]
) -> str:
    compact_schema = json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"{instructions.rstrip()}\n\n"
        "Return only one JSON object matching this JSON Schema exactly:\n"
        f"{compact_schema}"
    )


def _chat_user_content(
    *,
    input_text: str,
    input_text_blocks: tuple[ModelInputTextBlock, ...],
    enable_prompt_caching: bool,
) -> object:
    if not input_text_blocks:
        return input_text
    if "".join(block.text for block in input_text_blocks) != input_text:
        raise ValueError("model input blocks must reproduce input_text exactly")

    content: list[dict[str, object]] = []
    for block in input_text_blocks:
        rendered: dict[str, object] = {
            "type": "text",
            "text": block.text,
        }
        if block.cache_breakpoint and enable_prompt_caching:
            rendered["cache_control"] = {"type": "ephemeral"}
        content.append(rendered)
    return content


def _completion_message_text(response: object) -> str:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    message = _field(choices[0], "message")
    return _content_text(_field(message, "content"))


def _completion_delta_text(chunk: object) -> str:
    choices = _field(chunk, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    delta = _field(choices[0], "delta")
    return _content_text(_field(delta, "content"))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for part in content:
        text = _field(part, "text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _nonblank_string(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _token_count(value: object, name: str) -> int | None:
    count = _field(value, name)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def _chat_token_usage(response: object) -> ModelTokenUsage | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    input_tokens = _token_count(usage, "prompt_tokens")
    output_tokens = _token_count(usage, "completion_tokens")
    total_tokens = _token_count(usage, "total_tokens")
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None

    input_details = _field(usage, "prompt_tokens_details")
    output_details = _field(usage, "completion_tokens_details")
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
