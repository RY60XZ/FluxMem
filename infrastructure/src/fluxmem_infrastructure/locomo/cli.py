from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
        help="new or empty result directory",
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
            "judge": settings.answer_model,
            "extraction": settings.extraction_model,
            "reconciliation": settings.reconciliation_model,
            "lifecycle": settings.lifecycle_model,
            "embedding": settings.embedding_model,
        }
        with bootstrap_from_env(
            dotenv_path=arguments.dotenv,
            retrieval_settings=_LOCOMO_RETRIEVAL_SETTINGS,
            answer_with_history=False,
        ) as runtime:
            summary = LocomoRunner(
                memory=runtime.memory,
                answering=runtime.agent,
                judge=LLMLocomoJudge(
                    provider=runtime.model_provider,
                    settings=runtime.settings.answer_model_settings(),
                ),
                output_dir=arguments.output,
                dataset_path=data_path,
                dataset_sha256=dataset_sha256(data_path),
                mode="all" if arguments.all else "single",
                model_settings=model_settings,
                git_revision=_git_revision(),
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
