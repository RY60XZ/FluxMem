from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from fluxmem import HybridRetrievalSettings
from fluxmem_infrastructure.config import load_settings
from fluxmem_infrastructure.locomo.dataset import (
    DEFAULT_DATA_PATH,
    dataset_sha256,
    ensure_official_dataset,
    load_dataset,
    select_conversations,
)
from fluxmem_infrastructure.locomo.judge import LLMLocomoJudge
from fluxmem_infrastructure.locomo.prompts import (
    answer_prompt,
    full_context_answer_prompt,
)
from fluxmem_infrastructure.locomo.runner import LocomoRunner
from fluxmem_infrastructure.runtime import bootstrap_from_env


_LOCOMO_RETRIEVAL_SETTINGS = HybridRetrievalSettings(
    max_query_messages=1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluxmem-locomo",
        description="Run the LoCoMo QA benchmark against FluxMem.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run selected LoCoMo conversations")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="run the complete ten-conversation benchmark",
    )
    selection.add_argument(
        "--conversation",
        metavar="SAMPLE_ID",
        help="run one conversation by sample_id, such as conv-26",
    )
    run.add_argument(
        "--data",
        type=Path,
        help=(
            "LoCoMo JSON path; when omitted, download and verify the pinned "
            f"release at {DEFAULT_DATA_PATH}"
        ),
    )
    run.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new/empty result directory, or an existing run with --resume",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume the checkpoint in --output without repeating completed work",
    )
    run.add_argument(
        "--context-mode",
        choices=("memory", "full"),
        default="memory",
        help=(
            "answer from retrieved memories (memory) or the complete raw "
            "conversation without memory learning (full)"
        ),
    )
    run.add_argument(
        "--dotenv",
        type=Path,
        default=Path(".env"),
        help="runtime environment file (default: .env)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        data_path = (
            arguments.data
            if arguments.data is not None
            else ensure_official_dataset()
        )
        conversations = load_dataset(data_path)
        selected = select_conversations(
            conversations,
            run_all=arguments.all,
            sample_id=arguments.conversation,
        )
        settings = load_settings(dotenv_path=arguments.dotenv)
        model_settings = {
            "answer": settings.answer_model,
            "judge": settings.judge_model,
            "reranker": settings.reranker_model,
            "extraction": settings.extraction_model,
            "lifecycle": settings.lifecycle_model,
            "embedding": settings.embedding_model,
        }
        full_context = arguments.context_mode == "full"
        answer_context_settings = settings.answer_context_settings()
        if full_context:
            answer_context_settings = replace(
                answer_context_settings,
                maximum_history_messages=(
                    max(len(conversation.turns) for conversation in selected) + 1
                ),
            )
        with bootstrap_from_env(
            dotenv_path=arguments.dotenv,
            retrieval_settings=_LOCOMO_RETRIEVAL_SETTINGS,
            answer_context_settings=answer_context_settings,
            answer_with_history=full_context,
            answer_with_memories=not full_context,
            enable_memory_learning=not full_context,
            enable_memory_reranking=not full_context,
            answer_instructions=(
                full_context_answer_prompt() if full_context else answer_prompt()
            ),
        ) as runtime:
            summary = LocomoRunner(
                memory=runtime.memory,
                answering=runtime.agent,
                judge=LLMLocomoJudge(
                    provider=runtime.model_provider,
                    settings=runtime.settings.judge_model_settings(),
                ),
                output_dir=arguments.output,
                dataset_path=data_path,
                dataset_sha256=dataset_sha256(data_path),
                mode="all" if arguments.all else "single",
                model_settings=model_settings,
                context_mode=arguments.context_mode,
                git_revision=_git_revision(),
                resume=arguments.resume,
            ).run(selected)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"fluxmem-locomo: {error}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["conversations_with_errors"] else 0


def _git_revision() -> str | None:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{revision}-dirty" if dirty else revision


if __name__ == "__main__":
    sys.exit(main())
