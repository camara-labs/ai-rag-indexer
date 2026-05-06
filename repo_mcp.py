"""
Stage 3 — MCP server exposing semantic search over the repository index.

Reuses the same embedding client and Qdrant collection populated by
repo_embed.py. Exposes a single tool, `search_repositories`, that returns
the top-K most relevant repository summaries for a natural-language query.

Runs as a stdio MCP server — compatible with Claude Desktop, Cursor,
the MCP Inspector, or any MCP-capable client.

Configuration (.env):
    QDRANT_URL          — default http://localhost:6333
    QDRANT_API_KEY      — optional
    EMBED_MODEL         — model id used to embed the query
    REPOS_COLLECTION    — fallback collection name when --collection is omitted
    OPENAI_BASE_URL     — embedding server URL
    OPENAI_API_KEY      — any string for local servers

Usage:
    python repo_mcp.py --collection repos_index

Example client config (Claude Desktop, claude_desktop_config.json):
    {
      "mcpServers": {
        "repo-index": {
          "command": "python",
          "args": ["/abs/path/to/indexer/repo_mcp.py", "--collection", "repos_index"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from llm_client import DEFAULT_EMBED_MODEL, embed, make_client

load_dotenv()


def _make_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def _format_hit(hit) -> dict[str, Any]:
    payload = hit.payload or {}
    return {
        "score": float(hit.score),
        "repository": payload.get("symbol", ""),
        "path": payload.get("file_path", ""),
        "domain": payload.get("namespace", ""),
        "summary": payload.get("summary", ""),
        "main_technologies": payload.get("main_technologies") or [],
        "tags": payload.get("tags") or [],
    }


def search(
    query: str,
    collection: str,
    k: int = 5,
    embed_model: str | None = None,
) -> list[dict[str, Any]]:
    """Embed `query` and return the top-K matching repository summaries."""
    if not query or not query.strip():
        return []
    client = make_client()
    qdrant = _make_qdrant()
    model = embed_model or DEFAULT_EMBED_MODEL
    vector = embed(client, model, query.strip())
    hits = qdrant.search(
        collection_name=collection,
        query_vector=vector,
        limit=max(1, int(k)),
        with_payload=True,
    )
    return [_format_hit(h) for h in hits]


def build_server(default_collection: str, default_k: int, embed_model: str):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "error: the `mcp` package is required to run the MCP server.\n"
            "       pip install mcp"
        ) from exc

    server = FastMCP("repo-index")

    @server.tool()
    def search_repositories(query: str, k: int = default_k) -> list[dict[str, Any]]:
        """Return the top-K repository summaries most relevant to `query`.

        Args:
            query: Natural-language description of what the caller is looking for
                (e.g. "Python service that ingests events into Kafka").
            k: Number of results to return.
        """
        return search(query, collection=default_collection, k=k, embed_model=embed_model)

    return server


def main() -> int:
    ap = argparse.ArgumentParser(description="MCP server for repository semantic search.")
    ap.add_argument(
        "--collection",
        default=os.getenv("REPOS_COLLECTION"),
        help="Qdrant collection populated by repo_embed.py "
             "(default: REPOS_COLLECTION env)",
    )
    ap.add_argument("--k", type=int, default=5, help="Default number of results (default: 5)")
    ap.add_argument(
        "--embed-model",
        default=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL),
        dest="embed_model",
        help="Embedding model id (default: EMBED_MODEL from .env)",
    )
    args = ap.parse_args()

    if not args.collection:
        print("error: --collection is required (or set REPOS_COLLECTION env)", file=sys.stderr)
        return 1

    server = build_server(args.collection, args.k, args.embed_model)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
