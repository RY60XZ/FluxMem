from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from fluxmem.application.ports.llm import (
    ModelProviderError,
    ModelTimeoutError,
    StructuredModelResponse,
)


class OpenAIResponsesProvider:
    """Structured generation and text streaming through OpenAI Responses."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
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
        )

    def stream_text(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        timeout_seconds: float,
        maximum_output_tokens: int,
    ) -> Iterable[str]:
        """Yield user-facing text deltas without buffering the response."""

        stream = None
        received_text = False
        try:
            request_client = self._client
            if hasattr(request_client, "with_options"):
                request_client = request_client.with_options(
                    timeout=timeout_seconds
                )
            stream = request_client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=maximum_output_tokens,
                store=False,
                stream=True,
            )
            for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str) and delta:
                        received_text = True
                        yield delta
                elif event_type in {"error", "response.failed"}:
                    raise ModelProviderError(
                        _stream_error_message(event)
                    )
        except ModelProviderError:
            raise
        except TimeoutError as error:
            raise ModelTimeoutError("OpenAI response timed out") from error
        except Exception as error:
            if "timeout" in type(error).__name__.lower():
                raise ModelTimeoutError("OpenAI response timed out") from error
            raise ModelProviderError("OpenAI response request failed") from error
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if not received_text:
            raise ModelProviderError("OpenAI response contained no output text")


def _stream_error_message(event: object) -> str:
    error = getattr(event, "error", None)
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = "OpenAI streaming response failed"
    return message
