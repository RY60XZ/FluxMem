from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from fluxmem import LLMContextSettings, LLMTaskSettings, MemoryLayerSettings
from fluxmem_infrastructure.answering import (
    AnswerContextSettings,
    AnswerModelSettings,
)
from fluxmem_infrastructure.providers.openrouter import OPENROUTER_BASE_URL


DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"


@dataclass(frozen=True, slots=True)
class InfrastructureSettings:
    """Validated configuration for the example OpenRouter runtime."""

    database_url: str = field(repr=False)
    openrouter_api_key: str = field(repr=False)
    answer_model: str = DEFAULT_OPENROUTER_MODEL
    judge_model: str = DEFAULT_OPENROUTER_MODEL
    extraction_model: str = DEFAULT_OPENROUTER_MODEL
    lifecycle_model: str = DEFAULT_OPENROUTER_MODEL
    embedding_model: str = DEFAULT_OPENROUTER_EMBEDDING_MODEL
    openrouter_base_url: str = OPENROUTER_BASE_URL
    openrouter_http_referer: str | None = None
    openrouter_app_name: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_maximum_output_tokens: int = 2_048
    memory_history_messages: int = 128
    memory_write_history_messages: int = 20
    answer_history_messages: int = 128
    answer_input_token_budget: int = 64_000
    embedding_timeout_seconds: float = 30.0
    openrouter_strict_json_schema: bool = False
    openrouter_prompt_caching: bool = True
    enable_memory_extraction: bool = True
    enable_memory_writes: bool = True
    enable_llm_lifecycle: bool = True

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> InfrastructureSettings:
        values = os.environ if environ is None else environ
        shared_model = (
            _optional(values, "FLUXMEM_LLM_MODEL") or DEFAULT_OPENROUTER_MODEL
        )
        answer_model = _optional(values, "FLUXMEM_ANSWER_MODEL") or shared_model
        return cls(
            database_url=_required(values, "FLUXMEM_DATABASE_URL"),
            openrouter_api_key=_required(values, "OPENROUTER_API_KEY"),
            answer_model=answer_model,
            judge_model=(
                _optional(values, "FLUXMEM_JUDGE_MODEL") or answer_model
            ),
            extraction_model=(
                _optional(values, "FLUXMEM_EXTRACTION_MODEL") or shared_model
            ),
            lifecycle_model=(
                _optional(values, "FLUXMEM_LIFECYCLE_MODEL") or shared_model
            ),
            embedding_model=(
                _optional(values, "FLUXMEM_EMBEDDING_MODEL")
                or DEFAULT_OPENROUTER_EMBEDDING_MODEL
            ),
            openrouter_base_url=(
                _optional(values, "OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL
            ),
            openrouter_http_referer=_optional(
                values, "OPENROUTER_HTTP_REFERER"
            ),
            openrouter_app_name=_optional(values, "OPENROUTER_APP_NAME"),
            llm_timeout_seconds=_positive_float(
                values, "FLUXMEM_LLM_TIMEOUT_SECONDS", default=60.0
            ),
            llm_maximum_output_tokens=_positive_int(
                values, "FLUXMEM_LLM_MAX_OUTPUT_TOKENS", default=2_048
            ),
            memory_history_messages=_positive_int(
                values, "FLUXMEM_MEMORY_HISTORY_MESSAGES", default=128
            ),
            memory_write_history_messages=_positive_int(
                values,
                "FLUXMEM_MEMORY_WRITE_HISTORY_MESSAGES",
                default=20,
            ),
            answer_history_messages=_positive_int(
                values, "FLUXMEM_ANSWER_HISTORY_MESSAGES", default=128
            ),
            answer_input_token_budget=_positive_int(
                values, "FLUXMEM_ANSWER_INPUT_TOKEN_BUDGET", default=64_000
            ),
            embedding_timeout_seconds=_positive_float(
                values, "FLUXMEM_EMBEDDING_TIMEOUT_SECONDS", default=30.0
            ),
            openrouter_strict_json_schema=_boolean(
                values, "FLUXMEM_OPENROUTER_STRICT_JSON_SCHEMA", default=False
            ),
            openrouter_prompt_caching=_boolean(
                values, "FLUXMEM_OPENROUTER_PROMPT_CACHING", default=True
            ),
            enable_memory_extraction=_boolean(
                values, "FLUXMEM_ENABLE_MEMORY_EXTRACTION", default=True
            ),
            enable_memory_writes=_boolean(
                values, "FLUXMEM_ENABLE_MEMORY_WRITES", default=True
            ),
            enable_llm_lifecycle=_boolean(
                values, "FLUXMEM_ENABLE_LLM_LIFECYCLE", default=True
            ),
        )

    def task_settings(
        self,
        model: str,
        *,
        repair_invalid_output: bool = True,
    ) -> LLMTaskSettings:
        return LLMTaskSettings(
            model=model,
            timeout_seconds=self.llm_timeout_seconds,
            maximum_output_tokens=self.llm_maximum_output_tokens,
            repair_invalid_output=repair_invalid_output,
        )

    def memory_settings(self) -> MemoryLayerSettings:
        return MemoryLayerSettings(
            maximum_history_messages=self.memory_history_messages,
            enable_memory_extraction=self.enable_memory_extraction,
            enable_memory_writes=self.enable_memory_writes,
        )

    def memory_context_settings(self) -> LLMContextSettings:
        return LLMContextSettings(
            maximum_history_messages=self.memory_history_messages,
            maximum_write_history_messages=self.memory_write_history_messages,
        )

    def answer_model_settings(self) -> AnswerModelSettings:
        return AnswerModelSettings(
            model=self.answer_model,
            timeout_seconds=self.llm_timeout_seconds,
            maximum_output_tokens=self.llm_maximum_output_tokens,
        )

    def judge_model_settings(self) -> AnswerModelSettings:
        return AnswerModelSettings(
            model=self.judge_model,
            timeout_seconds=self.llm_timeout_seconds,
            maximum_output_tokens=self.llm_maximum_output_tokens,
        )

    def answer_context_settings(self) -> AnswerContextSettings:
        return AnswerContextSettings(
            maximum_history_messages=self.answer_history_messages,
            maximum_input_tokens=self.answer_input_token_budget,
        )


def load_settings(
    *,
    dotenv_path: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> InfrastructureSettings:
    combined: dict[str, str] = {}
    if dotenv_path is not None:
        path = Path(dotenv_path)
        if path.is_file():
            try:
                from dotenv import dotenv_values
            except ImportError as error:
                raise RuntimeError(
                    "Loading a .env file requires python-dotenv"
                ) from error
            combined.update(
                {
                    key: value
                    for key, value in dotenv_values(path).items()
                    if isinstance(value, str)
                }
            )
    combined.update(os.environ if environ is None else environ)
    return InfrastructureSettings.from_environment(combined)


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value.strip() if value is not None and value.strip() else None


def _required(environ: Mapping[str, str], name: str) -> str:
    value = _optional(environ, name)
    if value is None:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _positive_float(
    environ: Mapping[str, str], name: str, *, default: float
) -> float:
    raw = _optional(environ, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(
    environ: Mapping[str, str], name: str, *, default: int
) -> int:
    raw = _optional(environ, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(
    environ: Mapping[str, str], name: str, *, default: bool
) -> bool:
    raw = _optional(environ, name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
