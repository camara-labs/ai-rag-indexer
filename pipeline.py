"""
Orchestrates the full RAG indexing pipeline:

    detect_languages() / provided list
        → chunk_repo() per language
        → save JSONL
        → embed_chunks()
        → store_chunks()

Usage (programmatic):
    from pathlib import Path
    from pipeline import run

    jsonl_path = run(
        repo_path=Path("/path/to/repo"),
        collection="my_collection",
    )

Usage (standalone):
    python pipeline.py \\
        --repo-path ../codebase/my-app \\
        --collection my_app

    # Restrict to a single language:
    python pipeline.py \\
        --repo-path ../codebase/my-app \\
        --collection my_app \\
        --language typescript
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from chunkers import _SUPPORTED, chunk_repo as _chunk_repo, detect_languages
from embedder import embed_chunks
from storer import store_chunks


def run(
    repo_path: Path,
    collection: str,
    languages: list[str] | None = None,
    chunks_output: Path | None = None,
    embed_model: str | None = None,
    max_chunk_chars: int = 0,
    summarize: bool = False,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    summary_max_tokens: int = 256,
) -> Path:
    """
    Run the full indexing pipeline.

    Parameters
    ----------
    repo_path:
        Absolute path to the repository to index.
    collection:
        Qdrant collection name to upsert chunks into.
    languages:
        List of languages to index. If ``None`` (default), languages are
        auto-detected by scanning the repository for known file extensions.
    chunks_output:
        Where to save the intermediate JSONL. Defaults to
        ``.chunks/<repo_name>.jsonl`` relative to the indexer directory.
    embed_model:
        Embedding model id override. Defaults to ``EMBED_MODEL`` env var.
    max_chunk_chars:
        Drop chunks longer than this many characters. 0 = no limit.
    summarize:
        If True, run an LLM summarization step after chunking.
    llm_model:
        LLM model id for summarization. Defaults to ``LLM_MODEL`` env var.
    llm_base_url:
        LLM server base URL. Defaults to ``LLM_BASE_URL`` / ``OPENAI_BASE_URL``.
    summary_max_tokens:
        Max tokens per summary response. Default: 256.

    Returns
    -------
    Path
        Path to the saved chunks JSONL.
    """
    repo_path = Path(repo_path).resolve()

    # ── Language detection ────────────────────────────────────────────────────
    if languages is None:
        languages = detect_languages(repo_path)
        if not languages:
            raise ValueError(
                f"No supported source files found in {repo_path}.\n"
                f"Supported languages: {', '.join(_SUPPORTED)}"
            )
        print(f"  Auto-detected languages: {', '.join(languages)}")
    else:
        print(f"  Languages: {', '.join(languages)}")

    if chunks_output is None:
        chunks_dir = Path(__file__).parent / ".chunks"
        chunks_output = chunks_dir / f"{repo_path.name}.jsonl"

    total_steps = 4 if summarize else 3

    # ── Step 1: Chunking ──────────────────────────────────────────────────────
    print(f"\n[1/{total_steps}] Chunking {repo_path}...")
    all_chunks = []
    for lang in languages:
        try:
            lang_chunks = _chunk_repo(language=lang, repo_path=repo_path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise
        print(f"  [{lang}] {len(lang_chunks)} chunks")
        all_chunks.extend(lang_chunks)

    print(f"  Total: {len(all_chunks)} chunks")

    if max_chunk_chars > 0:
        before = len(all_chunks)
        all_chunks = [ch for ch in all_chunks if len(ch.code) <= max_chunk_chars]
        dropped = before - len(all_chunks)
        if dropped:
            print(f"  Filtered {dropped} chunk(s) exceeding {max_chunk_chars:,} chars")

    chunks_output.parent.mkdir(parents=True, exist_ok=True)
    raw_dicts = [asdict(ch) for ch in all_chunks]
    with chunks_output.open("w", encoding="utf-8") as fh:
        for ch in raw_dicts:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"  Saved -> {chunks_output}")

    # ── Step 2 (optional): Summarization ─────────────────────────────────────
    if summarize:
        from summarizer import summarize_chunks, DEFAULT_LLM_MODEL, DEFAULT_LLM_BASE_URL
        model = llm_model or DEFAULT_LLM_MODEL
        if not model:
            raise ValueError(
                "LLM model required for summarization. "
                "Set LLM_MODEL in .env or pass --llm-model."
            )
        url = llm_base_url or DEFAULT_LLM_BASE_URL
        print(f"\n[2/4] Summarizing {len(raw_dicts)} chunks (model={model})...")
        summarize_chunks(raw_dicts, model=model, base_url=url, max_tokens=summary_max_tokens)
        with chunks_output.open("w", encoding="utf-8") as fh:
            for ch in raw_dicts:
                fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
        print(f"  Updated -> {chunks_output}")

    # ── Step 3 (or 2): Embedding ──────────────────────────────────────────────
    embed_step = 3 if summarize else 2
    print(f"\n[{embed_step}/{total_steps}] Embedding chunks (model={embed_model or 'from .env'})...")
    embedded = embed_chunks(raw_dicts, model=embed_model)
    print(f"  Embedded {len(embedded)} chunks")

    # ── Step 4 (or 3): Store in Qdrant ───────────────────────────────────────
    store_step = 4 if summarize else 3
    print(f"\n[{store_step}/{total_steps}] Storing in Qdrant collection '{collection}'...")
    n = store_chunks(embedded, collection=collection)
    print(f"  Upserted {n} new points")

    return chunks_output


def main() -> int:
    ap = argparse.ArgumentParser(description="Full RAG indexing pipeline")
    ap.add_argument("--repo-path", type=Path, required=True, dest="repo_path",
                    help="Path to the repository to index")
    ap.add_argument("--collection", required=True,
                    help="Qdrant collection name")
    ap.add_argument("--language", default=None,
                    choices=list(_SUPPORTED),
                    help="Restrict to a single language (default: auto-detect all)")
    ap.add_argument("--chunks-output", type=Path, default=None, dest="chunks_output",
                    help="Path for the intermediate chunks JSONL")
    ap.add_argument("--embed-model", default=None, dest="embed_model",
                    help="Embedding model id override")
    ap.add_argument("--max-chunk-chars", type=int, default=0, metavar="N",
                    dest="max_chunk_chars",
                    help="Skip chunks larger than N chars. 0 = no limit. Suggested: 6000")
    ap.add_argument("--summarize", action="store_true", default=False,
                    help="Run LLM summarization step after chunking")
    ap.add_argument("--llm-model", default=None, dest="llm_model",
                    help="LLM model id for summarization (default: LLM_MODEL from .env)")
    ap.add_argument("--llm-base-url", default=None, dest="llm_base_url",
                    help="LLM server base URL (default: LLM_BASE_URL from .env)")
    ap.add_argument("--summary-max-tokens", type=int, default=256, dest="summary_max_tokens",
                    metavar="N",
                    help="Max tokens per summary response (default: 256)")
    args = ap.parse_args()

    languages = [args.language] if args.language else None

    try:
        jsonl = run(
            repo_path=args.repo_path,
            collection=args.collection,
            languages=languages,
            chunks_output=args.chunks_output,
            embed_model=args.embed_model,
            max_chunk_chars=args.max_chunk_chars,
            summarize=args.summarize,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            summary_max_tokens=args.summary_max_tokens,
        )
        print(f"\nPipeline complete. Chunks at: {jsonl}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
