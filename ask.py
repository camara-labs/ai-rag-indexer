"""
Etapa 3: pergunta -> retrieval no Qdrant -> prompt -> LLM (streaming)
via servidor OpenAI-compatível (LM Studio, Ollama, vLLM, llama.cpp).

Usage:
    python ask.py "como deletar um TodoItem do banco?" \\
        --llm-model qwen3-8b-instruct \\
        --embed-model text-embedding-bge-m3

Descubra os ids dos modelos no seu servidor:
    curl http://localhost:1234/v1/models
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from llm_client import embed, make_client

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_SYSTEM_PROMPTS: dict[str, str] = {
    "csharp": (
        "Você é um assistente especialista em C# e .NET que responde perguntas sobre o código fornecido como contexto.\n\n"
        "Regras estritas:\n"
        "1. Use APENAS os trechos de código fornecidos abaixo. Não invente nomes de classes, métodos ou arquivos.\n"
        "2. Se a resposta não estiver nos trechos, diga exatamente: \"Não encontrado nos trechos fornecidos.\"\n"
        "3. Sempre cite o arquivo e o intervalo de linhas ao referenciar código, no formato `caminho/arquivo.cs:linha`.\n"
        "4. Quando relevante, mostre o nome completo do método ou classe (ex.: `DeleteTodoItemCommandHandler.Handle`).\n"
        "5. Responda em português, de forma técnica e direta."
    ),
    "typescript": (
        "Você é um assistente especialista em TypeScript e Node.js/browser que responde perguntas sobre o código fornecido como contexto.\n\n"
        "Regras estritas:\n"
        "1. Use APENAS os trechos de código fornecidos abaixo. Não invente nomes de funções, classes, interfaces ou arquivos.\n"
        "2. Se a resposta não estiver nos trechos, diga exatamente: \"Não encontrado nos trechos fornecidos.\"\n"
        "3. Sempre cite o arquivo e o intervalo de linhas ao referenciar código, no formato `caminho/arquivo.ts:linha`.\n"
        "4. Quando relevante, mostre o nome completo da função, classe ou tipo (ex.: `UserService.findById`).\n"
        "5. Responda em português, de forma técnica e direta."
    ),
    "javascript": (
        "Você é um assistente especialista em JavaScript que responde perguntas sobre o código fornecido como contexto.\n\n"
        "Regras estritas:\n"
        "1. Use APENAS os trechos de código fornecidos abaixo. Não invente nomes de funções, classes ou arquivos.\n"
        "2. Se a resposta não estiver nos trechos, diga exatamente: \"Não encontrado nos trechos fornecidos.\"\n"
        "3. Sempre cite o arquivo e o intervalo de linhas ao referenciar código, no formato `caminho/arquivo.js:linha`.\n"
        "4. Quando relevante, mostre o nome completo da função ou classe (ex.: `UserService.findById`).\n"
        "5. Responda em português, de forma técnica e direta."
    ),
    "terraform": (
        "Você é um assistente especialista em Terraform e infraestrutura como código (IaC) que responde perguntas sobre a configuração HCL fornecida como contexto.\n\n"
        "Regras estritas:\n"
        "1. Use APENAS os trechos de configuração fornecidos abaixo. Não invente nomes de recursos, módulos ou variáveis.\n"
        "2. Se a resposta não estiver nos trechos, diga exatamente: \"Não encontrado nos trechos fornecidos.\"\n"
        "3. Sempre cite o arquivo ao referenciar configuração, no formato `caminho/arquivo.tf:linha`.\n"
        "4. Quando relevante, mostre o identificador completo do recurso (ex.: `aws_s3_bucket.my_bucket`).\n"
        "5. Responda em português, de forma técnica e direta."
    ),
}

# Fallback para chunks sem campo language (indexados antes da migração)
_DEFAULT_SYSTEM_PROMPT = _SYSTEM_PROMPTS["csharp"]


def _detect_language(hits, override: str | None = None) -> str:
    if override:
        return override.lower().strip()
    for hit in hits:
        lang = (hit.payload or {}).get("language", "")
        if lang:
            return lang
    return "csharp"


def _estimate_tokens(text: str) -> int:
    """Estimativa rápida: ~4 caracteres por token."""
    return len(text) // 4


def build_prompt(question: str, hits, max_ctx: int | None = None) -> tuple[str, int]:
    overhead = _estimate_tokens(SYSTEM_PROMPT + question + "\n\n---\nPergunta: \n\nResposta:")
    budget = (max_ctx - overhead) if max_ctx else None

    blocks = []
    used = 0
    skipped = 0
    for i, hit in enumerate(hits, 1):
        p = hit.payload or {}
        header = (
            f"[Trecho {i}] {p.get('file_path')}:{p.get('start_line')}-{p.get('end_line')}  "
            f"(símbolo: {p.get('symbol')}, tipo: {p.get('kind')})"
        )
        lang = p.get("language") or "csharp"
        block = f"{header}\n```{lang}\n{p.get('code', '')}\n```"
        cost = _estimate_tokens(block)
        if budget is not None and used + cost > budget:
            skipped += 1
            continue
        blocks.append(block)
        used += cost

    if skipped:
        print(f"[aviso] {skipped} chunk(s) descartado(s) para caber no contexto ({max_ctx} tokens)",
              file=sys.stderr)

    context = "\n\n".join(blocks)
    return f"{context}\n\n---\nPergunta: {question}\n\nResposta:", overhead + used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--collection", default=None,
                    help="Nome da collection no Qdrant (obrigatório)")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "text-embedding-bge-m3"))
    ap.add_argument("--embed-url", default=os.getenv("OPENAI_BASE_URL"),
                    help="URL do servidor de embeddings. Default: OPENAI_BASE_URL do .env")
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "google/gemma-4-e4b"),
                    help="Id do LLM no servidor (curl /v1/models para descobrir)")
    ap.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"),
                    help="URL do servidor de chat/LLM. Default: LLM_BASE_URL do .env")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-ctx", type=int, default=3500,
                    help="Limite estimado de tokens do prompt (padrão: 3500). Use 0 para desabilitar.")
    ap.add_argument("--show-context", action="store_true",
                    help="Imprime os chunks recuperados antes da resposta")
    ap.add_argument("--language", default=None,
                    choices=["csharp", "typescript", "javascript", "terraform"],
                    help="Força o system prompt para uma linguagem específica (padrão: inferido dos chunks)")
    args = ap.parse_args()

    if not args.collection:
        args.collection = input("Collection (Qdrant): ").strip()

    qdrant = QdrantClient(url=QDRANT_URL)
    embed_client = make_client(args.embed_url)
    llm_client = make_client(args.base_url)

    vector = embed(embed_client, args.embed_model, args.question)
    hits = qdrant.search(
        collection_name=args.collection,
        query_vector=vector,
        limit=args.k,
        with_payload=True,
    )

    if not hits:
        print("Nenhum trecho encontrado.", file=sys.stderr)
        return 1

    if args.show_context:
        print("=== Trechos recuperados ===")
        for i, h in enumerate(hits, 1):
            p = h.payload or {}
            print(f"#{i} score={h.score:.3f}  {p.get('symbol')}  "
                  f"({p.get('file_path')}:{p.get('start_line')}-{p.get('end_line')})")
        print("===========================\n")

    language = _detect_language(hits, args.language)
    system_prompt = _SYSTEM_PROMPTS.get(language, _DEFAULT_SYSTEM_PROMPT)

    max_ctx = args.max_ctx or None
    user_prompt, estimated_tokens = build_prompt(args.question, hits, max_ctx)
    if max_ctx:
        print(f"[info] tokens estimados no prompt: ~{estimated_tokens}", file=sys.stderr)

    stream = llm_client.chat.completions.create(
        model=args.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        temperature=args.temperature,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
