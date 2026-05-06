"""
Sumarizador de chunks via LLM.

Lê um arquivo JSONL de chunks (saída do chunker.py), chama um LLM para gerar
um campo `summary` para cada chunk e grava o resultado em outro JSONL.

Configuração via .env (ou variáveis de ambiente):
    OPENAI_BASE_URL   — URL base do servidor de LLM
    OPENAI_API_KEY    — chave de API (qualquer string para servidores locais)
    LLM_MODEL         — id do modelo de chat (ex: qwen3-4b, llama-3.1-8b)

Uso:
    python summarizer.py .chunks/clean-arch.jsonl
    python summarizer.py .chunks/clean-arch.jsonl --output .chunks/clean-arch-summarized.jsonl
    python summarizer.py .chunks/clean-arch.jsonl --model qwen3-4b --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import make_client

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

_SYSTEM_PROMPT = (
    "You are a code documentation assistant. "
    "Given a code snippet, write a concise 1-3 sentence summary of what it does. "
    "Focus on: purpose, key inputs/outputs, and any notable side effects. "
    "Be technical and precise. Reply with the summary only — no preamble."
)

_USER_TEMPLATE = """\
{kind} `{symbol}` — {namespace}
Signature: {signature}
File: {file_path}

```{language}
{code}
```
"""


def _build_user_message(chunk: dict) -> str:
    return _USER_TEMPLATE.format(
        kind=chunk.get("kind", "symbol"),
        symbol=chunk.get("symbol", "?"),
        namespace=chunk.get("namespace", ""),
        signature=chunk.get("signature", ""),
        file_path=chunk.get("file_path", ""),
        language=chunk.get("language") or "csharp",
        code=chunk.get("code", ""),
    )


def summarize_chunks(
    chunks: list[dict],
    model: str,
    base_url: str | None = None,
    skip_existing: bool = False,
    max_tokens: int = 256,
    already_done: set[str] | None = None,
    out_file=None,
) -> list[dict]:
    """Adiciona o campo `summary` a cada chunk usando o LLM.

    Args:
        chunks: lista de dicts de chunks (modificada in-place e retornada).
        model: id do modelo de chat.
        base_url: URL base do servidor de chat (usa LLM_BASE_URL/.env se None).
        skip_existing: se True, pula chunks que já têm `summary`.
        max_tokens: limite de tokens na resposta do LLM.
        already_done: set de content_hash já presentes no output (checkpoint).
        out_file: arquivo de saída aberto para escrita incremental.

    Returns:
        A mesma lista com `summary` preenchido.
    """
    client = make_client(base_url or DEFAULT_LLM_BASE_URL)
    skipped = 0
    errors = 0

    for chunk in tqdm(chunks, desc="Summarizing", unit="chunk"):
        if already_done and chunk.get("content_hash") in already_done:
            skipped += 1
            continue
        if skip_existing and chunk.get("summary"):
            skipped += 1
            continue
        if "Test" in chunk.get("file_path", ""):
            skipped += 1
            continue

        user_msg = _build_user_message(chunk)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                # Thinking models (e.g. Gemma) may exhaust max_tokens on reasoning;
                # fall back to reasoning_content so at least something is stored.
                content = (getattr(msg, "reasoning_content", None) or "").strip()
            chunk["summary"] = content

            timings = (resp.model_extra or {}).get("timings")
            if timings:
                tps = timings.get("predicted_per_second")
                if tps is not None:
                    tqdm.write(f"  {chunk.get('symbol', '?')}: {tps:.1f} tok/s")
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  [WARN] {chunk.get('symbol', '?')}: {exc}")
            chunk["summary"] = ""
            errors += 1

        if out_file is not None:
            out_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            out_file.flush()

    if skipped:
        print(f"  Skipped {skipped} already-summarized chunks.")
    if errors:
        print(f"  {errors} chunk(s) failed — summary left empty.")

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gera sumários LLM para cada chunk de um JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("jsonl", help="Arquivo JSONL de entrada (saída do chunker.py)")
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Arquivo JSONL de saída. "
            "Padrão: mesmo nome com sufixo -summarized.jsonl"
        ),
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL or None,
        help=(
            "ID do modelo de chat. "
            "Padrão: variável LLM_MODEL do .env"
        ),
    )
    ap.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help=(
            "URL base do servidor de chat. "
            "Padrão: LLM_BASE_URL do .env (ou OPENAI_BASE_URL se ausente)"
        ),
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        dest="max_tokens",
        help="Limite de tokens na resposta do LLM por chunk (padrão: 4096; aumente para modelos thinking)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip_existing",
        help="Não reprocessa chunks que já possuem campo `summary`",
    )
    args = ap.parse_args()

    # Validar modelo
    model: str = args.model or ""
    if not model:
        print(
            "Erro: nenhum modelo de chat configurado.\n"
            "  Defina LLM_MODEL no arquivo .env ou use --model <nome>.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Caminhos
    input_path = Path(args.jsonl)
    if not input_path.exists():
        print(f"Erro: arquivo não encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        if stem.endswith("-summarized"):
            output_path = input_path
        else:
            output_path = input_path.with_name(stem + "-summarized.jsonl")

    # Carregar
    with input_path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    base_url = args.base_url or DEFAULT_LLM_BASE_URL
    print(f"Carregados {len(chunks):,} chunks de {input_path}")
    print(f"Modelo   : {model}")
    print(f"Servidor : {base_url}")
    print(f"Saída    : {output_path}")

    # Carregar checkpoint: hashes já processados no output anterior
    already_done: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        h = entry.get("content_hash")
                        if h and entry.get("summary") is not None:
                            already_done.add(h)
                    except json.JSONDecodeError:
                        pass
        if already_done:
            print(f"Retomando: {len(already_done):,}/{len(chunks):,} chunks já processados em {output_path}")

    # Sumarizar e gravar incrementalmente
    with output_path.open("a", encoding="utf-8") as out_file:
        summarize_chunks(
            chunks,
            model=model,
            base_url=base_url,
            skip_existing=args.skip_existing,
            max_tokens=args.max_tokens,
            already_done=already_done,
            out_file=out_file,
        )

    total_with_summary = len(already_done) + sum(1 for c in chunks if c.get("summary") is not None and c.get("content_hash") not in already_done)
    print(f"\nPronto. {total_with_summary:,}/{len(chunks):,} chunks com sumário → {output_path}")


if __name__ == "__main__":
    main()
