from __future__ import annotations

from importlib.resources import files


_PROMPT_PACKAGE = "fluxmem.application.llm.prompts"
_PROMPT_NAMES = frozenset(
    {
        "lifecycle_evaluation",
        "memory_extraction",
        "memory_reconciliation",
        "repair",
    }
)


def load_prompt(name: str) -> str:
    """Load one packaged prompt without allowing arbitrary resource traversal."""

    if name not in _PROMPT_NAMES:
        raise ValueError(f"unknown LLM prompt: {name}")
    prompt = (
        files(_PROMPT_PACKAGE)
        .joinpath(f"{name}.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not prompt:
        raise ValueError(f"LLM prompt is blank: {name}")
    return prompt


def render_repair_prompt(*, validation_error: str) -> str:
    template = load_prompt("repair")
    placeholder = "{validation_error}"
    if template.count(placeholder) != 1:
        raise ValueError("repair prompt must contain one validation_error placeholder")
    return template.replace(placeholder, validation_error)
