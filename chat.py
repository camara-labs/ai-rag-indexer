"""
Chat interativo com contexto de sessão persistente.
Mantém histórico de perguntas e respostas entre turnos.

Usage:
    # Nova sessão
    python chat.py \\
        --collection minha-collection \\
        --llm-model qwen3-8b-instruct \\
        --embed-model text-embedding-bge-m3

    # Primeira pergunta já na linha de comando
    python chat.py "como deletar um TodoItem?" --collection minha-collection

    # Restaurar sessão existente
    python chat.py --session 20260430143022-como-deletar-todo-item \\
        --collection minha-collection

Descubra os ids dos modelos no seu servidor:
    curl http://localhost:1234/v1/models

Sessões ficam em: indexer/sessions/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from llm_client import embed, make_client

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
SESSIONS_DIR = Path(__file__).parent / "sessions"

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre um código C# fornecido como contexto.

Regras estritas:
1. Use APENAS os trechos de código fornecidos abaixo. Não invente nomes de classes, métodos ou arquivos.
2. Se a resposta não estiver nos trechos, diga exatamente: "Não encontrado nos trechos fornecidos."
3. Sempre cite o arquivo e o intervalo de linhas ao referenciar código, no formato `caminho/arquivo.cs:linha`.
4. Quando relevante, mostre o nome completo do método ou classe (ex.: `DeleteTodoItemCommandHandler.Handle`).
5. Responda em português, de forma técnica e direta.
6. Você tem acesso ao histórico completo da conversa — use-o para entender o contexto das perguntas anteriores.
"""


def _slugify(text: str, max_words: int = 6) -> str:
    """Normaliza as primeiras palavras do texto em slug para compor o nome da sessão."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = nfkd.encode("ascii", "ignore").decode()
    ascii_text = re.sub(r"[^a-z0-9\s]", "", ascii_text)
    words = ascii_text.split()[:max_words]
    slug = "-".join(words)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "sessao"


def _new_session_id(first_question: str) -> str:
    now = datetime.now()
    # yyyyMMddHHmmssff  (ff = centiseconds: 2 dígitos)
    ts = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 10000:02d}"
    slug = _slugify(first_question)
    return f"{ts}-{slug}"


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_session(session_id: str) -> dict:
    """Carrega sessão por ID exato ou por prefixo único."""
    path = _session_path(session_id)
    if not path.exists():
        matches = sorted(SESSIONS_DIR.glob(f"{session_id}*.json"))
        if len(matches) == 1:
            path = matches[0]
        elif len(matches) > 1:
            print("Múltiplas sessões encontradas:", file=sys.stderr)
            for m in matches:
                print(f"  {m.stem}", file=sys.stderr)
            raise SystemExit("Especifique um ID mais preciso.")
        else:
            raise SystemExit(f"Sessão não encontrada: {session_id}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Garante que o campo messages inclui o system prompt atualizado
    if not data.get("messages") or data["messages"][0].get("role") != "system":
        data["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + data.get("messages", [])

    return data


def _save_session(session: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session["id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def _new_session(session_id: str) -> dict:
    return {
        "id": session_id,
        "created_at": datetime.now().isoformat(),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "exchanges": [],
    }


def _print_session_history(session: dict) -> None:
    exchanges = session.get("exchanges", [])
    if not exchanges:
        return
    print(f"\n=== Histórico ({len(exchanges)} troca(s)) ===", file=sys.stderr)
    for i, ex in enumerate(exchanges, 1):
        q = ex["question"]
        a = ex["answer"]
        print(f"\n[{i}] você> {q}", file=sys.stderr)
        preview = a[:200].replace("\n", " ")
        ellipsis = "..." if len(a) > 200 else ""
        print(f"     assistente> {preview}{ellipsis}", file=sys.stderr)
    print("=== Fim do histórico ===\n", file=sys.stderr)


def _run_spinner(stop_event: threading.Event) -> None:
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        print(f"\r{frames[i % len(frames)]} aguardando resposta...", end="", flush=True)
        stop_event.wait(0.1)
        i += 1
    print("\r" + " " * 25 + "\r", end="", flush=True)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def build_user_message(question: str, hits) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        p = hit.payload or {}
        header = (
            f"[Trecho {i}] {p.get('file_path')}:{p.get('start_line')}-{p.get('end_line')}  "
            f"(símbolo: {p.get('symbol')}, tipo: {p.get('kind')})"
        )
        blocks.append(f"{header}\n```csharp\n{p.get('code', '')}\n```")

    context = "\n\n".join(blocks)
    return f"{context}\n\n---\nPergunta: {question}\n\nResposta:"


def chat_turn(
    session: dict,
    question: str,
    qdrant: QdrantClient,
    embed_client,
    llm_client,
    args,
) -> str:
    vector = embed(embed_client, args.embed_model, question)
    hits = qdrant.search(
        collection_name=args.collection,
        query_vector=vector,
        limit=args.k,
        with_payload=True,
    )

    if not hits:
        print("Nenhum trecho encontrado.", file=sys.stderr)
        return ""

    if args.show_context:
        print("=== Trechos recuperados ===")
        for i, h in enumerate(hits, 1):
            p = h.payload or {}
            print(
                f"#{i} score={h.score:.3f}  {p.get('symbol')}  "
                f"({p.get('file_path')}:{p.get('start_line')}-{p.get('end_line')})"
            )
        print("===========================\n")

    user_content = build_user_message(question, hits)
    session["messages"].append({"role": "user", "content": user_content})

    ctx_tokens = sum(_estimate_tokens(m.get("content", "")) for m in session["messages"])

    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_run_spinner, args=(stop_spinner,), daemon=True)
    spinner.start()

    stream = llm_client.chat.completions.create(
        model=args.llm_model,
        messages=session["messages"],
        stream=True,
        temperature=args.temperature,
    )

    answer_parts: list[str] = []
    t_start: float | None = None
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if t_start is None:
                stop_spinner.set()
                spinner.join()
                print("\nassistente> ", end="", flush=True)
                t_start = time.perf_counter()
            print(delta, end="", flush=True)
            answer_parts.append(delta)

    stop_spinner.set()
    spinner.join()
    t_elapsed = time.perf_counter() - t_start if t_start else 0
    print()

    answer = "".join(answer_parts)
    out_tokens = _estimate_tokens(answer)
    tps = out_tokens / t_elapsed if t_elapsed > 0 else 0
    print(
        f"[✓ {out_tokens} tok gerados | {tps:.1f} tok/s | contexto: ~{ctx_tokens} tok]",
        file=sys.stderr,
    )
    session["messages"].append({"role": "assistant", "content": answer})
    session["exchanges"].append(
        {
            "question": question,
            "answer": answer,
            "hits": [
                {
                    "score": h.score,
                    "file_path": (h.payload or {}).get("file_path"),
                    "start_line": (h.payload or {}).get("start_line"),
                    "end_line": (h.payload or {}).get("end_line"),
                    "symbol": (h.payload or {}).get("symbol"),
                }
                for h in hits
            ],
        }
    )
    return answer


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat RAG com histórico de sessão persistente.")
    ap.add_argument(
        "question", nargs="?", default=None,
        help="Primeira pergunta (opcional; pode digitar no loop interativo)",
    )
    ap.add_argument(
        "--session", default=None,
        help="ID (ou prefixo) de uma sessão anterior para restaurar",
    )
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
    ap.add_argument("--show-context", action="store_true",
                    help="Imprime os chunks recuperados antes de cada resposta")
    args = ap.parse_args()

    if not args.collection:
        args.collection = input("Collection (Qdrant): ").strip()

    qdrant = QdrantClient(url=QDRANT_URL)
    embed_client = make_client(args.embed_url)
    llm_client = make_client(args.base_url)

    session: dict | None = None

    if args.session:
        session = load_session(args.session)
        n = len(session["exchanges"])
        print(f"[sessão restaurada] {session['id']}", file=sys.stderr)
        _print_session_history(session)
        print(f"[histórico] {n} troca(s) carregada(s). Continue a conversa abaixo.\n",
              file=sys.stderr)

    print("Digite sua pergunta (ou 'sair' / Ctrl+C para encerrar).\n", file=sys.stderr)

    first_question = args.question

    try:
        while True:
            if first_question is not None:
                question = first_question.strip()
                first_question = None
            else:
                try:
                    question = input("você> ").strip()
                except EOFError:
                    break

            if not question:
                continue
            if question.lower() in ("sair", "exit", "quit"):
                break

            if session is None:
                session_id = _new_session_id(question)
                session = _new_session(session_id)
                print(f"[sessão iniciada] {session_id}", file=sys.stderr)

            chat_turn(session, question, qdrant, embed_client, llm_client, args)
            _save_session(session)

    except KeyboardInterrupt:
        print("\n[interrompido]", file=sys.stderr)

    if session:
        _save_session(session)
        print(f"[sessão salva] {session['id']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
