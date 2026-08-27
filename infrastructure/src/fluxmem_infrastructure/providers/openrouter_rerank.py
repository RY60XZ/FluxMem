from __future__ import annotations

import os
from collections.abc import Mapping
from math import isfinite
from typing import Any

from fluxmem import ModelProviderError, ModelTimeoutError
from fluxmem_infrastructure.answering.reranker import (
    RerankProviderResponse,
    RerankProviderResult,
)
from fluxmem_infrastructure.providers.openrouter import OPENROUTER_BASE_URL


class OpenRouterRerankProvider:
    """Call OpenRouter's native cross-encoder rerank endpoint."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        http_referer: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self._owns_client = client is None
        if client is not None:
            self._client = client
            return
        try:
            import httpx
        except ImportError as error:
            raise RuntimeError(
                "OpenRouter rerank adapter requires httpx"
            ) from error

        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if resolved_api_key is None or not resolved_api_key.strip():
            raise ValueError("OPENROUTER_API_KEY cannot be blank")
        if not base_url.strip():
            raise ValueError("OpenRouter base URL cannot be blank")
        headers = {
            "Authorization": f"Bearer {resolved_api_key.strip()}",
            "Content-Type": "application/json",
        }
        if http_referer is not None and http_referer.strip():
            headers["HTTP-Referer"] = http_referer.strip()
        if app_name is not None and app_name.strip():
            headers["X-OpenRouter-Title"] = app_name.strip()
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/",
            headers=headers,
        )

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: tuple[str, ...],
        top_n: int,
        timeout_seconds: float,
    ) -> RerankProviderResponse:
        if not model.strip() or not query.strip():
            raise ValueError("rerank model and query cannot be blank")
        if not documents or top_n < 1 or top_n > len(documents):
            raise ValueError("rerank document bounds are invalid")
        if timeout_seconds <= 0:
            raise ValueError("rerank timeout must be positive")
        try:
            response = self._client.post(
                "rerank",
                json={
                    "model": model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": top_n,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            if "timeout" in type(error).__name__.lower():
                raise ModelTimeoutError(
                    "OpenRouter rerank request timed out"
                ) from error
            raise ModelProviderError(
                "OpenRouter rerank request failed"
            ) from error
        try:
            return _parse_response(payload=payload, fallback_model=model)
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                "OpenRouter rerank response was invalid"
            ) from error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def _parse_response(
    *,
    payload: object,
    fallback_model: str,
) -> RerankProviderResponse:
    if not isinstance(payload, Mapping):
        raise TypeError("rerank response must be an object")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TypeError("rerank response results must be an array")
    results: list[RerankProviderResult] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise TypeError("rerank result must be an object")
        index = _nonnegative_int(raw.get("index"), field="result index")
        score = _finite_float(
            raw.get("relevance_score"),
            field="relevance score",
        )
        results.append(
            RerankProviderResult(index=index, relevance_score=score)
        )
    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise TypeError("rerank usage must be an object")
    model = payload.get("model")
    provider = payload.get("provider")
    response_id = payload.get("id")
    return RerankProviderResponse(
        results=tuple(results),
        model=(
            model
            if isinstance(model, str) and model.strip()
            else fallback_model
        ),
        response_id=str(response_id) if response_id else None,
        provider=(
            provider
            if isinstance(provider, str) and provider.strip()
            else None
        ),
        total_tokens=(
            _optional_nonnegative_int(usage.get("total_tokens"))
            if usage is not None
            else None
        ),
        search_units=(
            _optional_nonnegative_int(usage.get("search_units"))
            if usage is not None
            else None
        ),
    )


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field="usage value")


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
