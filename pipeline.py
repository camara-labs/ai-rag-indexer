"""
Orchestrates the full RAG indexing pipeline:

    chunk_repo()  →  save JSONL  →  embed_chunks()  →  store_chunks()

Usage (programmatic):
    from pathlib import Path
    from pipeline import run

    jsonl_path = run(
        language="csharp",
        repo_path=Path("/path/to/repo"),
        collection="my_collection",
    )

Usage (standalone):
    python pipeline.py \\
        --language csharp \\
        --repo-path ../codebase/CleanArchitecture \\
        --collection csharp_clean_arch
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from chunkers import chunk_repo as _chunk_repo
from embedder import embed_chunks
from storer import store_chunks


def run(
    language: str,
    repo_path: Path,
    collection: str,
    chunks_output: Path | None = None,
    embed_model: str | None = None,
    max_chunk_chars: int = 0,
) -> Path:
    """
    Run the full indexing pipeline.

    Parameters
    ----------
    language:
        Source language to parse (e.g. ``"csharp"``).
    repo_path:
        Absolute path to the repository to index.
    collection:
        Qdrant collection name to upsert chunks into.
    chunks_output:
        Where to save the intermediate JSONL. Defaults to
        ``chunks/<language>-<repo_name>.jsonl`` relative to the indexer dir.
    embed_model:
        Embedding model id override. Defaults to ``EMBED_MODEL`` env var.

    Returns
    -------
    Path
        Path to the saved chunks JSONL.
    """
    repo_path = Path(repo_path).resolve()

    if chunks_output is None:
        chunks_dir = Path(__file__).parent / "chunks"
        chunks_output = chunks_dir / f"{language}-{repo_path.name}.jsonl"

    # ── Step 1: Chunking ──────────────────────────────────────────────────────
    print(f"\n[1/3] Chunking {repo_path} (language={language})...")
    try:
        chunks = _chunk_repo(language=language, repo_path=repo_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise

    print(f"  Produced {len(chunks)} chunks")

    if max_chunk_chars > 0:
        before = len(chunks)
        chunks = [ch for ch in chunks if len(ch.code) <= max_chunk_chars]
        dropped = before - len(chunks)
        if dropped:
            print(f"  Filtered {dropped} chunk(s) exceeding {max_chunk_chars:,} chars")

    chunks_output.parent.mkdir(parents=True, exist_ok=True)
    with chunks_output.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    print(f"  Saved -> {chunks_output}")

    # ── Step 2: Embedding ─────────────────────────────────────────────────────
    print(f"\n[2/3] Embedding chunks (model={embed_model or 'from .env'})...")
    raw_dicts = [asdict(ch) for ch in chunks]
    embedded = embed_chunks(raw_dicts, model=embed_model)
    print(f"  Embedded {len(embedded)} chunks")

    # ── Step 3: Store in Qdrant ───────────────────────────────────────────────
    print(f"\n[3/3] Storing in Qdrant collection '{collection}'...")
    n = store_chunks(embedded, collection=collection)
    print(f"  Upserted {n} new points")

    return chunks_output


def main() -> int:
    ap = argparse.ArgumentParser(description="Full RAG indexing pipeline")
    ap.add_argument("--language", default="csharp",
                    help="Source language (default: csharp)")
    ap.add_argument("--repo-path", type=Path, required=True,
                    help="Path to the repository to index")
    ap.add_argument("--collection", required=True,
                    help="Qdrant collection name")
    ap.add_argument("--chunks-output", type=Path, default=None,
                    help="Path for the intermediate chunks JSONL")
    ap.add_argument("--embed-model", default=None,
                    help="Embedding model id override")
    ap.add_argument("--max-chunk-chars", type=int, default=0, metavar="N",
                    help="Skip chunks larger than N chars. 0 = no limit. Suggested: 6000")
    args = ap.parse_args()

    try:
        jsonl = run(
            language=args.language,
            repo_path=args.repo_path,
            collection=args.collection,
            chunks_output=args.chunks_output,
            embed_model=args.embed_model,
            max_chunk_chars=args.max_chunk_chars,
        )
        print(f"\nPipeline complete. Chunks at: {jsonl}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
