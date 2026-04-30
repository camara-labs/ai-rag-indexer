"""
Etapa 2: lê chunks.jsonl, gera embeddings via servidor OpenAI-compatível e
faz upsert no Qdrant.

Usage:
    python index_chunks.py <chunks.jsonl> \\
        --collection csharp_code_bgem3_lms \\
        --embed-model text-embedding-bge-m3

Pré-requisitos:
    - Qdrant rodando em localhost:6333 (docker compose up -d)
    - Um servidor OpenAI-compatível rodando com um modelo de embedding carregado:
        - LM Studio:  aba "Local Server" -> carregue bge-m3 -> Start Server
        - Ollama:     ollama pull bge-m3 (e passe --base-url http://localhost:11434/v1)
    - Descubra o id exato do modelo com:
        curl http://localhost:1234/v1/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from llm_client import embed, make_client

QDRANT_URL = "http://localhost:6333"


def ensure_collection(qdrant: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        return
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Índices de payload para filtros rápidos
    for field, schema in [
        ("file_path", qm.PayloadSchemaType.KEYWORD),
        ("namespace", qm.PayloadSchemaType.KEYWORD),
        ("kind", qm.PayloadSchemaType.KEYWORD),
        ("symbol", qm.PayloadSchemaType.KEYWORD),
        ("content_hash", qm.PayloadSchemaType.KEYWORD),
    ]:
        qdrant.create_payload_index(collection_name=name, field_name=field, field_schema=schema)


def existing_hashes(qdrant: QdrantClient, collection: str) -> set[str]:
    """Retorna o conjunto de content_hash já indexados (para indexação incremental)."""
    hashes: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = qdrant.scroll(
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


def stable_id(content_hash: str) -> int:
    """Converte hash em inteiro estável (Qdrant aceita int ou UUID como ID)."""
    return int(content_hash[:16], 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--collection", default="csharp_code_bgem3_lms")
    ap.add_argument("--embed-model", default="text-embedding-bge-m3",
                    help="Id do modelo no servidor (curl /v1/models para descobrir)")
    ap.add_argument("--base-url", default=None,
                    help="URL do servidor OpenAI-compat. Default: http://localhost:1234/v1 (LM Studio)")
    ap.add_argument("--batch", type=int, default=64, help="Pontos por upsert")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} não existe", file=sys.stderr)
        return 1

    qdrant = QdrantClient(url=QDRANT_URL)
    client = make_client(args.base_url)

    print(f"Sondando dimensão do modelo {args.embed_model}...")
    probe = embed(client, args.embed_model, "probe")
    dim = len(probe)
    print(f"  dim = {dim}")

    ensure_collection(qdrant, args.collection, dim)

    print("Lendo hashes já indexados...")
    seen = existing_hashes(qdrant, args.collection)
    print(f"  {len(seen)} chunks já no Qdrant")

    chunks = [json.loads(l) for l in args.jsonl.open(encoding="utf-8")]
    to_index = [c for c in chunks if c["content_hash"] not in seen]
    print(f"Total: {len(chunks)} chunks no JSONL, {len(to_index)} novos para indexar")

    if not to_index:
        print("Nada novo. Saindo.")
        return 0

    buffer: list[qm.PointStruct] = []
    for chunk in tqdm(to_index, desc="Embedding"):
        try:
            vector = embed(client, args.embed_model, chunk["code"])
        except Exception as exc:
            print(f"warn: falha ao embedar {chunk['symbol']}: {exc}", file=sys.stderr)
            continue

        buffer.append(
            qm.PointStruct(
                id=stable_id(chunk["content_hash"]),
                vector=vector,
                payload=chunk,
            )
        )
        if len(buffer) >= args.batch:
            qdrant.upsert(collection_name=args.collection, points=buffer)
            buffer.clear()

    if buffer:
        qdrant.upsert(collection_name=args.collection, points=buffer)

    info = qdrant.get_collection(args.collection)
    print(f"Concluído. Coleção '{args.collection}' agora tem {info.points_count} pontos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
