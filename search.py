"""
Etapa 2: busca semântica no Qdrant via servidor OpenAI-compatível (LM Studio/Ollama).

Usage:
    python search.py "como deletar um TodoItem?" \\
        --collection csharp_code_bgem3_lms \\
        --embed-model text-embedding-bge-m3
"""

from __future__ import annotations

import argparse

from qdrant_client import QdrantClient

from llm_client import embed, make_client

QDRANT_URL = "http://localhost:6333"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--collection", default="csharp_code_bgem3_lms")
    ap.add_argument("--embed-model", default="text-embedding-bge-m3")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--kind", default=None, help="Filtrar por kind (method, class, ...)")
    args = ap.parse_args()

    qdrant = QdrantClient(url=QDRANT_URL)
    client = make_client(args.base_url)

    vector = embed(client, args.embed_model, args.query)

    query_filter = None
    if args.kind:
        from qdrant_client.http import models as qm

        query_filter = qm.Filter(
            must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value=args.kind))]
        )

    results = qdrant.search(
        collection_name=args.collection,
        query_vector=vector,
        limit=args.k,
        query_filter=query_filter,
        with_payload=True,
    )

    print(f"\nQuery: {args.query}\n")
    for i, r in enumerate(results, 1):
        p = r.payload or {}
        print(f"#{i}  score={r.score:.4f}  {p.get('kind')}  {p.get('symbol')}")
        print(f"    {p.get('file_path')}:{p.get('start_line')}-{p.get('end_line')}")
        print(f"    namespace: {p.get('namespace')}")
        sig = p.get("signature", "").replace("\n", " ").replace("\r", "")
        print(f"    signature: {sig[:120]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
