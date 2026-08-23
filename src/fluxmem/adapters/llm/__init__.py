"""Optional external model-provider adapters."""

from fluxmem.adapters.llm.openai_responses import OpenAIResponsesProvider
from fluxmem.adapters.llm.openrouter_chat import (
    OPENROUTER_BASE_URL,
    OpenRouterChatCompletionsProvider,
)

__all__ = (
    "OPENROUTER_BASE_URL",
    "OpenAIResponsesProvider",
    "OpenRouterChatCompletionsProvider",
)
