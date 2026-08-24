from __future__ import annotations

import os
from typing import Any

from fluxmem import EMBEDDING_DIMENSIONS, Embedding, EmbeddingProviderError
from fluxmem_infrastructure.providers.openrouter import OPENROUTER_BASE_URL


class OpenRouterEmbeddingProvider:
    """Generate FluxMem vectors through OpenRouter's Embeddings endpoint."""

    def __init__(
        self,
        *,
        model: str = "openai/text-embedding-3-small",
        dimensions: int = EMBEDDING_DIMENSIONS,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        http_referer: str | None = None,
        app_name: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model cannot be blank")
        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"FluxMem currently requires {EMBEDDING_DIMENSIONS} dimensions"
            )
        if timeout_seconds <= 0:
            raise ValueError("embedding timeout must be positive")
        self.model = model
        self.dimensions = dimensions
        self._timeout_seconds = timeout_seconds
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenRouter embedding adapter requires openai"
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

    def embed(self, *, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("embedding input cannot be blank")
        try:
            request_client = self._client
            if hasattr(request_client, "with_options"):
                request_client = request_client.with_options(
                    timeout=self._timeout_seconds
                )
            response = request_client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            data = getattr(response, "data", None)
            if not isinstance(data, (list, tuple)) or not data:
                raise ValueError("embedding response contained no vectors")
            raw_values = getattr(data[0], "embedding", None)
            if not isinstance(raw_values, (list, tuple)):
                raise ValueError("embedding response contained an invalid vector")
            values = tuple(float(value) for value in raw_values)
            if len(values) != self.dimensions:
                raise ValueError(
                    f"embedding response returned {len(values)} dimensions; "
                    f"expected {self.dimensions}"
                )
            response_model = getattr(response, "model", None)
            vector_model = (
                response_model
                if isinstance(response_model, str) and response_model.strip()
                else self.model
            )
            return Embedding(values=values, model=vector_model)
        except EmbeddingProviderError:
            raise
        except Exception as error:
            raise EmbeddingProviderError(
                "OpenRouter embedding request failed"
            ) from error
