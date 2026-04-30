"""
Interactive CLI for the RAG indexing pipeline.

Any parameter not provided as a flag is asked interactively via questionary.

Usage (fully interactive):
    python cli.py

Usage (fully scriptable, no prompts):
    python cli.py \\
        --language csharp \\
        --repo-path /abs/path/to/repo \\
        --collection my_collection

Usage (mixed — only missing flags are prompted):
    python cli.py --language csharp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Supported languages (shown as choices in the interactive prompt) ──────────
_SUPPORTED_LANGUAGES = ["csharp"]


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

    if args.language is None:
        args.language = questionary.select(
            "Source language:",
            choices=_SUPPORTED_LANGUAGES,
        ).ask()
        if args.language is None:
            sys.exit(0)

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
        default_jsonl = f"chunks/{args.collection}.jsonl"
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


def _confirm(args: argparse.Namespace) -> bool:
    """Print a summary and ask for confirmation."""
    try:
        import questionary
    except ImportError:
        return True

    print()
    print("  Language   :", args.language)
    print("  Repo path  :", args.repo_path)
    print("  Collection :", args.collection)
    print("  JSONL path :", args.chunks_output or "(auto)")
    print("  Embed model:", args.embed_model or "(from .env)")
    print("  Max chunk  :", f"{args.max_chunk_chars:,} chars" if args.max_chunk_chars else "(no limit)")
    print()

    return questionary.confirm("Proceed with indexing?", default=True).ask() or False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive CLI for the RAG indexing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any missing flag will be asked interactively.\n"
            "All settings are read from .env (OPENAI_BASE_URL, EMBED_MODEL, QDRANT_URL, ...)."
        ),
    )
    ap.add_argument(
        "--language",
        choices=_SUPPORTED_LANGUAGES,
        default=None,
        help="Source language to parse",
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

    # Confirmation (skipped if --yes or if all args were provided via flags)
    if not args.yes and not _confirm(args):
        print("Aborted.")
        return 0

    from pipeline import run

    try:
        jsonl = run(
            language=args.language,
            repo_path=args.repo_path,
            collection=args.collection,
            chunks_output=args.chunks_output,
            embed_model=args.embed_model,
            max_chunk_chars=args.max_chunk_chars,
        )
        print(f"\nPipeline complete. Chunks saved at: {jsonl}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
