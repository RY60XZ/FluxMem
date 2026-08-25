from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from fluxmem import (
    FluxMem,
    HybridRetrievalSettings,
    LLMContextSettings,
    LLMLifecycleEvaluator,
    LLMMemoryExtractor,
    LLMMemoryReconciler,
    StructuredModelProvider,
    bootstrap,
)
from fluxmem_infrastructure.answering import (
    AnswerContextSettings,
    AnswerGenerator,
    AnsweringAgent,
)
from fluxmem_infrastructure.config import InfrastructureSettings, load_settings
from fluxmem_infrastructure.providers import (
    OpenRouterChatCompletionsProvider,
    OpenRouterEmbeddingProvider,
)


@dataclass(frozen=True, slots=True)
class AnsweringRuntime:
    """Example composition of FluxMem with an answering agent."""

    memory: FluxMem
    agent: AnsweringAgent
    model_provider: StructuredModelProvider
    settings: InfrastructureSettings

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.memory.close()


def bootstrap_from_env(
    *,
    dotenv_path: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
    memory_context_settings: LLMContextSettings | None = None,
    retrieval_settings: HybridRetrievalSettings | None = None,
    answer_context_settings: AnswerContextSettings | None = None,
    answer_with_history: bool = True,
    answer_with_memories: bool = True,
    enable_memory_learning: bool = True,
    answer_instructions: str | None = None,
    **engine_options: object,
) -> AnsweringRuntime:
    settings = load_settings(dotenv_path=dotenv_path, environ=environ)
    provider = OpenRouterChatCompletionsProvider(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        http_referer=settings.openrouter_http_referer,
        app_name=settings.openrouter_app_name,
        strict_json_schema=settings.openrouter_strict_json_schema,
        enable_prompt_caching=settings.openrouter_prompt_caching,
    )
    embedding_provider = OpenRouterEmbeddingProvider(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.embedding_model,
        timeout_seconds=settings.embedding_timeout_seconds,
        http_referer=settings.openrouter_http_referer,
        app_name=settings.openrouter_app_name,
    )
    context_settings = (
        memory_context_settings or settings.memory_context_settings()
    )
    memory = bootstrap(
        database_url=settings.database_url,
        memory_extractor=(
            LLMMemoryExtractor(
                provider=provider,
                settings=settings.task_settings(settings.extraction_model),
                context_settings=context_settings,
            )
            if enable_memory_learning
            else None
        ),
        memory_reconciler=(
            LLMMemoryReconciler(
                provider=provider,
                settings=settings.task_settings(settings.reconciliation_model),
                context_settings=context_settings,
            )
            if enable_memory_learning
            else None
        ),
        lifecycle_evaluator=(
            LLMLifecycleEvaluator(
                provider=provider,
                settings=settings.task_settings(
                    settings.lifecycle_model,
                    repair_invalid_output=False,
                ),
            )
            if settings.enable_llm_lifecycle and enable_memory_learning
            else None
        ),
        embedding_provider=embedding_provider,
        retrieval_settings=retrieval_settings,
        settings=settings.memory_settings(),
        **engine_options,
    )
    generator = AnswerGenerator(
        provider=provider,
        settings=settings.answer_model_settings(),
        context_settings=(
            answer_context_settings or settings.answer_context_settings()
        ),
        instructions=answer_instructions,
    )
    return AnsweringRuntime(
        memory=memory,
        agent=AnsweringAgent(
            memory=memory,
            generator=generator,
            include_history=answer_with_history,
            include_memories=answer_with_memories,
        ),
        model_provider=provider,
        settings=settings,
    )
