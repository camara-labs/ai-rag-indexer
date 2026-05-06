"""
Interactive CLI for the RAG indexing pipeline.

Languages are detected automatically by scanning the repository for known
file extensions. Any parameter not provided as a flag is asked interactively.

Usage (fully interactive):
    python cli.py

Usage (fully scriptable, no prompts):
    python cli.py \\
        --repo-path /abs/path/to/repo \\
        --collection my_collection

Usage (restrict to one language):
    python cli.py \\
        --repo-path /abs/path/to/repo \\
        --collection my_collection \\
        --language typescript
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chunkers import _SUPPORTED


def _ask_missing(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any missing values interactively using questionary."""
    try:
        import questionary
    except ImportError:
        print(
            "error: 'questionary' is not installed. "
            "Run: pip install questionary",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.repo_path is None:
        raw = questionary.path(
            "Absolute path to the repository:",
            only_directories=True,
        ).ask()
        if raw is None:
            sys.exit(0)
        args.repo_path = Path(raw)

    if args.collection is None:
        args.collection = questionary.text(
            "Qdrant collection name:",
            validate=lambda v: bool(v.strip()) or "Collection name cannot be empty",
        ).ask()
        if args.collection is None:
            sys.exit(0)

    if args.chunks_output is None:
        default_jsonl = f".chunks/{args.collection}.jsonl"
        raw = questionary.text(
            "Intermediate JSONL output path:",
            default=default_jsonl,
            instruction="(Enter to accept default based on collection name)",
        ).ask()
        if raw is None:
            sys.exit(0)
        args.chunks_output = Path(raw.strip())

    if args.embed_model is None:
        from llm_client import DEFAULT_EMBED_MODEL
        raw = questionary.text(
            "Embedding model id (leave blank to use .env / default):",
            default="",
            instruction=f"Current default: {DEFAULT_EMBED_MODEL}",
        ).ask()
        if raw is None:
            sys.exit(0)
        args.embed_model = raw.strip() or None

    return args


def _ask_summarize(args: argparse.Namespace) -> argparse.Namespace:
    """Ask whether to run the summarization step, and which model to use."""
    try:
        import questionary
    except ImportError:
        return args

    if args.summarize is None:
        answer = questionary.confirm(
            "Run summarization step? (adds LLM summaries to chunks before embedding)",
            default=False,
        ).ask()
        if answer is None:
            sys.exit(0)
        args.summarize = answer

    if args.summarize and args.llm_model is None:
        from summarizer import DEFAULT_LLM_MODEL
        instruction = f"Current default: {DEFAULT_LLM_MODEL}" if DEFAULT_LLM_MODEL else "(required — set LLM_MODEL in .env or enter below)"
        raw = questionary.text(
            "LLM model id for summarization:",
            default=DEFAULT_LLM_MODEL or "",
            instruction=instruction,
        ).ask()
        if raw is None:
            sys.exit(0)
        args.llm_model = raw.strip() or DEFAULT_LLM_MODEL or None

    return args


def _confirm(args: argparse.Namespace, detected_languages: list[str]) -> bool:
    """Print a summary and ask for confirmation."""
    try:
        import questionary
    except ImportError:
        return True

    lang_label = ", ".join(detected_languages)
    if args.language:
        lang_label += " (specified)"
    else:
        lang_label += " (auto-detected)"

    summarize_label = "no"
    if args.summarize:
        summarize_label = f"yes (model={args.llm_model or '(from .env)'})"

    print()
    print("  Languages  :", lang_label)
    print("  Repo path  :", args.repo_path)
    print("  Collection :", args.collection)
    print("  JSONL path :", args.chunks_output or "(auto)")
    print("  Embed model:", args.embed_model or "(from .env)")
    print("  Max chunk  :", f"{args.max_chunk_chars:,} chars" if args.max_chunk_chars else "(no limit)")
    print("  Summarize  :", summarize_label)
    print()

    return questionary.confirm("Proceed with indexing?", default=True).ask() or False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive CLI for the RAG indexing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Languages are auto-detected from the repository contents.\n"
            "Use --language to restrict to a single language.\n"
            "All settings are read from .env (OPENAI_BASE_URL, EMBED_MODEL, QDRANT_URL, ...)."
        ),
    )
    ap.add_argument(
        "--repo-path",
        type=Path,
        default=None,
        dest="repo_path",
        help="Absolute path to the repository",
    )
    ap.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name",
    )
    ap.add_argument(
        "--language",
        choices=list(_SUPPORTED),
        default=None,
        help="Restrict indexing to a single language (default: auto-detect all)",
    )
    ap.add_argument(
        "--chunks-output",
        type=Path,
        default=None,
        dest="chunks_output",
        help="Path for the intermediate chunks JSONL (optional)",
    )
    ap.add_argument(
        "--embed-model",
        default=None,
        dest="embed_model",
        help="Embedding model id override (optional, reads .env by default)",
    )
    ap.add_argument(
        "--max-chunk-chars",
        type=int,
        default=0,
        dest="max_chunk_chars",
        metavar="N",
        help="Skip chunks larger than N chars. 0 = no limit. Suggested: 6000",
    )
    ap.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="summarize",
        help="Run LLM summarization step after chunking (default: ask interactively)",
    )
    ap.add_argument(
        "--llm-model",
        default=None,
        dest="llm_model",
        help="LLM model id for summarization (default: LLM_MODEL from .env)",
    )
    ap.add_argument(
        "--llm-base-url",
        default=None,
        dest="llm_base_url",
        help="LLM server base URL for summarization (default: LLM_BASE_URL from .env)",
    )
    ap.add_argument(
        "--summary-max-tokens",
        type=int,
        default=256,
        dest="summary_max_tokens",
        metavar="N",
        help="Max tokens per summary response (default: 256)",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = ap.parse_args()

    # Fill any missing args interactively
    args = _ask_missing(args)

    # Validate repo path
    if not args.repo_path.is_dir():
        print(f"error: {args.repo_path} is not a directory", file=sys.stderr)
        return 1

    # Resolve languages: explicit flag or auto-detect
    if args.language:
        languages = [args.language]
    else:
        from chunkers import detect_languages
        languages = detect_languages(args.repo_path)
        if not languages:
            print(
                f"error: no supported source files found in {args.repo_path}\n"
                f"  Supported: {', '.join(_SUPPORTED)}",
                file=sys.stderr,
            )
            return 1

    # Ask about summarization (skipped if --yes or --summarize/--no-summarize passed)
    if not args.yes:
        args = _ask_summarize(args)
    elif args.summarize is None:
        args.summarize = False

    # Confirmation
    if not args.yes and not _confirm(args, languages):
        print("Aborted.")
        return 0

    from pipeline import run

    try:
        jsonl = run(
            repo_path=args.repo_path,
            collection=args.collection,
            languages=languages,
            chunks_output=args.chunks_output,
            embed_model=args.embed_model,
            max_chunk_chars=args.max_chunk_chars,
            summarize=args.summarize or False,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            summary_max_tokens=args.summary_max_tokens,
        )
        print(f"\nPipeline complete. Chunks saved at: {jsonl}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
