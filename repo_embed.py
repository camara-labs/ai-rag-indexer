"""
Stage 2 — Embed and persist repository summaries.

Reads the JSON produced by repo_summarize.py, formats each repository as a
single semantic unit, and feeds it through the *existing* embedding and
persistence pipeline:

    embedder.embed_chunks(...)   -> adds the "vector" field
    storer.store_chunks(...)     -> upserts into Qdrant (incremental)

This script intentionally does not re-implement embeddings or storage —
it only adapts the new repo-summary data shape into the chunk-dict format
that the existing pipeline already understands. Each repository is treated
as one chunk with kind="repository".

Usage:
    python repo_embed.py .repos/repos.json --collection repos_index
    python repo_embed.py .repos/             --collection repos_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from embedder import embed_chunks
from storer import store_chunks


def _load_summaries(input_path: Path) -> list[dict]:
    """Accept either a JSON array file, a single-object JSON file, or a
    directory containing one JSON file per repository."""
    if input_path.is_dir():
        items: list[dict] = []
        for f in sorted(input_path.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"warn: skipping {f.name}: {exc}", file=sys.stderr)
                continue
            if isinstance(data, list):
                items.extend(data)
            elif isinstance(data, dict):
                items.append(data)
        return items

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unexpected JSON root type in {input_path}: {type(data).__name__}")


def _build_embedding_text(summary: dict) -> str:
    """Compose the text that drives retrieval for this repository."""
    parts: list[str] = [f"Repository: {summary.get('repository', '')}"]
    if summary.get("domain"):
        parts.append(f"Domain: {summary['domain']}")
    techs = summary.get("main_technologies") or []
    if techs:
        parts.append("Technologies: " + ", ".join(techs))
    tags = summary.get("tags") or []
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    if summary.get("summary"):
        parts.append("")
        parts.append(summary["summary"])
    return "\n".join(parts)


def to_chunk(summary: dict) -> dict:
    """Adapt a repo summary to the chunk-dict shape expected by embedder/storer."""
    code = _build_embedding_text(summary)
    name = summary.get("repository", "")
    path = summary.get("path", "")

    content_hash = summary.get("content_hash")
    if not content_hash:
        blob = code + "|" + path
        content_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    return {
        "code": code,
        "kind": "repository",
        "symbol": name,
        "namespace": summary.get("domain", "") or "",
        "file_path": path,
        "language": "markdown",
        "summary": summary.get("summary", ""),
        "main_technologies": summary.get("main_technologies") or [],
        "tags": summary.get("tags") or [],
        "content_hash": content_hash,
        # Placeholders so the chunk dict matches the existing payload shape.
        "start_line": 0,
        "end_line": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed and persist repository summaries.")
    ap.add_argument(
        "input", type=Path,
        help="repos.json file (array or single object) or directory of *.json files",
    )
    ap.add_argument("--collection", required=True, help="Qdrant collection name")
    ap.add_argument(
        "--embed-model", default=None, dest="embed_model",
        help="Embedding model id (default: EMBED_MODEL from .env)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 1

    try:
        summaries = _load_summaries(args.input)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not summaries:
        print("No summaries found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(summaries)} repository summar{'y' if len(summaries) == 1 else 'ies'}.")
    chunks = [to_chunk(s) for s in summaries]

    embedded = embed_chunks(chunks, model=args.embed_model)
    if not embedded:
        print("No chunks produced — nothing to store.", file=sys.stderr)
        return 1

    n = store_chunks(embedded, collection=args.collection)
    print(f"Done. {n} new repositor{'y' if n == 1 else 'ies'} indexed in '{args.collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
