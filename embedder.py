"""
Embedding step — generates vector embeddings for a list of chunks.

Reads EMBED_MODEL, OPENAI_BASE_URL, and OPENAI_API_KEY from the .env file
(via llm_client). The model can also be overridden at call time.

Usage (standalone):
    python embedder.py chunks/clean-arch.jsonl --output chunks/clean-arch-embedded.jsonl

Programmatic:
    from embedder import embed_chunks
    enriched = embed_chunks(chunks, model="text-embedding-bge-m3")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from llm_client import DEFAULT_EMBED_MODEL, embed, make_client


def embed_chunks(
    chunks: list[dict],
    model: str | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """
    Add a ``vector`` field to each chunk dict by calling the embedding server.

    Parameters
    ----------
    chunks:
        List of chunk dicts (as produced by the chunking step).
    model:
        Embedding model id. Falls back to ``EMBED_MODEL`` env var /
        ``DEFAULT_EMBED_MODEL`` from llm_client.
    batch_size:
        Number of chunks to embed before logging progress. Not used for
        batching API calls (each call is individual to match server limits).

    Returns
    -------
    list[dict]
        Same chunks with an added ``"vector"`` key. Chunks that fail to embed
        are skipped (a warning is printed to stderr).
    """
    effective_model = model or DEFAULT_EMBED_MODEL
    client = make_client()

    enriched: list[dict] = []
    failed = 0

    for chunk in tqdm(chunks, desc=f"Embedding [{effective_model}]"):
        try:
            vector = embed(client, effective_model, chunk["code"])
        except Exception as exc:
            print(
                f"warn: failed to embed {chunk.get('symbol', '?')}: {exc}",
                file=sys.stderr,
            )
            failed += 1
            continue
        enriched.append({**chunk, "vector": vector})

    if failed:
        print(f"warn: {failed} chunk(s) skipped due to embedding errors", file=sys.stderr)

    return enriched


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed chunks and write enriched JSONL")
    ap.add_argument("jsonl", type=Path, help="Input chunks JSONL")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSONL path (default: overwrite input)")
    ap.add_argument("--model", default=None,
                    help=f"Embedding model id (default: EMBED_MODEL env / {DEFAULT_EMBED_MODEL})")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    chunks = [json.loads(line) for line in args.jsonl.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks from {args.jsonl}")

    enriched = embed_chunks(chunks, model=args.model)

    out_path = args.output or args.jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for ch in enriched:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Wrote {len(enriched)} embedded chunks -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
