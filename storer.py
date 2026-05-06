"""
Storage step — upserts embedded chunks into a Qdrant collection.

Reads QDRANT_URL and (optionally) QDRANT_API_KEY from the .env file.
The collection name is always passed explicitly.

Usage (standalone):
    python storer.py .chunks/clean-arch-embedded.jsonl --collection csharp_clean_arch

Programmatic:
    from storer import store_chunks
    store_chunks(embedded_chunks, collection="csharp_clean_arch")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

load_dotenv()

_DEFAULT_QDRANT_URL = "http://localhost:6333"
_UPSERT_BATCH = 64


def _make_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL)
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    for field, schema in [
        ("file_path", qm.PayloadSchemaType.KEYWORD),
        ("namespace", qm.PayloadSchemaType.KEYWORD),
        ("kind", qm.PayloadSchemaType.KEYWORD),
        ("symbol", qm.PayloadSchemaType.KEYWORD),
        ("content_hash", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema
        )


def _existing_hashes(client: QdrantClient, collection: str) -> set[str]:
    hashes: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=512,
            with_payload=["content_hash"],
            with_vectors=False,
            offset=next_offset,
        )
        for p in points:
            h = (p.payload or {}).get("content_hash")
            if h:
                hashes.add(h)
        if next_offset is None:
            break
    return hashes


def _stable_id(content_hash: str) -> int:
    return int(content_hash[:16], 16)


def store_chunks(
    chunks: list[dict],
    collection: str,
    batch_size: int = _UPSERT_BATCH,
) -> int:
    """
    Upsert embedded chunks into the Qdrant collection.

    Parameters
    ----------
    chunks:
        Chunks with a ``"vector"`` field (as produced by embedder.embed_chunks).
    collection:
        Target Qdrant collection name. Created automatically if it doesn't exist.
    batch_size:
        Number of points per upsert call.

    Returns
    -------
    int
        Number of new points upserted.

    Raises
    ------
    ValueError
        If any chunk is missing the ``"vector"`` field.
    """
    if not chunks:
        return 0

    if "vector" not in chunks[0]:
        raise ValueError(
            "Chunks must have a 'vector' field. Run embed_chunks() first."
        )

    qdrant = _make_qdrant_client()
    dim = len(chunks[0]["vector"])

    _ensure_collection(qdrant, collection, dim)

    print(f"Reading existing hashes from '{collection}'...")
    seen = _existing_hashes(qdrant, collection)
    print(f"  {len(seen)} chunks already indexed")

    to_store = [c for c in chunks if c.get("content_hash") not in seen]
    print(f"  {len(to_store)} new chunks to store")

    if not to_store:
        return 0

    buffer: list[qm.PointStruct] = []
    upserted = 0

    for chunk in tqdm(to_store, desc=f"Upserting -> '{collection}'"):
        payload = {k: v for k, v in chunk.items() if k != "vector"}
        buffer.append(
            qm.PointStruct(
                id=_stable_id(chunk["content_hash"]),
                vector=chunk["vector"],
                payload=payload,
            )
        )
        if len(buffer) >= batch_size:
            qdrant.upsert(collection_name=collection, points=buffer)
            upserted += len(buffer)
            buffer.clear()

    if buffer:
        qdrant.upsert(collection_name=collection, points=buffer)
        upserted += len(buffer)

    info = qdrant.get_collection(collection)
    print(f"Done. Collection '{collection}' now has {info.points_count} points.")
    return upserted


def main() -> int:
    ap = argparse.ArgumentParser(description="Store embedded chunks in Qdrant")
    ap.add_argument("jsonl", type=Path, help="Embedded chunks JSONL (must contain 'vector' field)")
    ap.add_argument("--collection", required=True, help="Qdrant collection name")
    ap.add_argument("--batch", type=int, default=_UPSERT_BATCH, help="Points per upsert call")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    chunks = [json.loads(line) for line in args.jsonl.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks from {args.jsonl}")

    try:
        n = store_chunks(chunks, collection=args.collection, batch_size=args.batch)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Upserted {n} new points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
