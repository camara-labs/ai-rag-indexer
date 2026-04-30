"""
Semantic chunker CLI — thin wrapper around the chunkers package.

Languages are auto-detected from the repository contents. Use --language
to restrict to a single language when needed.

Usage:
    python chunker.py <source_dir> <output.jsonl>
    python chunker.py <source_dir> <output.jsonl> --language typescript

Flags:
    --language            Restrict to one language (default: auto-detect all).
    --max-chunk-chars N   Skip chunks larger than N characters (default: no limit).
                          Suggested threshold: 6000. Chunks above this are almost
                          always generated code, migration files, or lookup tables.
    --top-large N         After chunking, print the N largest chunks as a sanity
                          check (default: 5). Use 0 to disable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from chunkers import _SUPPORTED, chunk_repo, detect_languages

_WARN_THRESHOLD = 6_000   # chars — used for the warning marker in the list


def _print_large_chunks(chunks, top_n: int) -> None:
    if top_n <= 0 or not chunks:
        return
    sorted_chunks = sorted(chunks, key=lambda c: len(c.code), reverse=True)[:top_n]
    print(f"\nTop {top_n} largest chunks (inspect for generated/noise code):")
    print(f"  {'chars':>7}  {'kind':<12}  {'symbol':<40}  file")
    print(f"  {'─'*7}  {'─'*12}  {'─'*40}  {'─'*40}")
    for ch in sorted_chunks:
        n = len(ch.code)
        marker = " ⚠" if n > _WARN_THRESHOLD else ""
        print(f"  {n:>7,}  {ch.kind:<12}  {ch.symbol:<40}  {ch.file_path}{marker}")
    if any(len(ch.code) > _WARN_THRESHOLD for ch in sorted_chunks):
        print(f"\n  ⚠  Chunks above {_WARN_THRESHOLD:,} chars are likely generated/noise.")
        print(f"     Re-run with --max-chunk-chars {_WARN_THRESHOLD} to exclude them.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Semantic chunker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("source_dir", type=Path, help="Directory containing source files")
    ap.add_argument("output", type=Path, help="Output JSONL file")
    ap.add_argument(
        "--language",
        default=None,
        choices=list(_SUPPORTED),
        help="Restrict to one language (default: auto-detect all)",
    )
    ap.add_argument(
        "--max-chunk-chars",
        type=int,
        default=0,
        metavar="N",
        help="Skip chunks larger than N characters. 0 = no limit. Suggested: 6000",
    )
    ap.add_argument(
        "--top-large",
        type=int,
        default=5,
        metavar="N",
        help="Print the N largest chunks after chunking for inspection (0 = disable)",
    )
    args = ap.parse_args()

    if not args.source_dir.is_dir():
        print(f"error: {args.source_dir} is not a directory", file=sys.stderr)
        return 1

    if args.language:
        languages = [args.language]
    else:
        languages = detect_languages(args.source_dir)
        if not languages:
            print(
                f"error: no supported source files found in {args.source_dir}\n"
                f"  Supported: {', '.join(_SUPPORTED)}",
                file=sys.stderr,
            )
            return 1
        print(f"Auto-detected languages: {', '.join(languages)}")

    chunks = []
    for lang in languages:
        try:
            lang_chunks = chunk_repo(language=lang, repo_path=args.source_dir)
            print(f"  [{lang}] {len(lang_chunks)} chunks")
            chunks.extend(lang_chunks)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Show large-chunk report before filtering (so the user sees what would be dropped)
    _print_large_chunks(chunks, args.top_large)

    # Optional size filter
    if args.max_chunk_chars > 0:
        before = len(chunks)
        chunks = [ch for ch in chunks if len(ch.code) <= args.max_chunk_chars]
        dropped = before - len(chunks)
        if dropped:
            print(f"\nFiltered {dropped} chunk(s) exceeding {args.max_chunk_chars:,} chars.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for ch in chunks:
            out.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    print(f"\nProduced {len(chunks)} chunks -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

