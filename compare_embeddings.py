"""
Compara dois modelos de embedding indexando os mesmos chunks em coleções
separadas do Qdrant e avaliando a qualidade de recuperação.

Métricas calculadas
───────────────────
• dim             – dimensionalidade do vetor
• index_time      – tempo total de embedding (s)
• score_mean/std  – média e desvio dos scores nos top-k resultados
• score_gap       – score[rank1] − score[rank_k]  (discriminação)
• rank_corr       – correlação de Spearman entre os rankings dos dois modelos

Modo "somente busca" (coleções já existentes):
    python compare_embeddings.py <chunks.jsonl> --skip-indexing

Usage completo:
    python compare_embeddings.py .chunks/clean-arch.jsonl \\
        --model-a text-embedding-bge-m3  \\
        --model-b qwen3-embedding         \\
        --col-a  clean_arch_bgem3         \\
        --col-b  clean_arch_qwen3         \\
        --sample 200                      \\
        --queries queries.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from llm_client import embed, make_client

QDRANT_URL = "http://localhost:6333"


# ── Verificação de modelos ─────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def verify_distinct_models(client_a, client_b, model_a: str, model_b: str) -> None:
    """Verifica se os dois modelos realmente retornam vetores distintos.

    Emite aviso (e instrução de correção) se o servidor ignorar o parâmetro
    `model` e servir o mesmo embedding para ambos os nomes.
    """
    print("\n[verificação] Testando se os modelos são distintos...")

    # Lista modelos disponíveis no servidor (melhor esforço)
    try:
        available = [m.id for m in client_a.models.list().data]
        print(f"  Modelos disponíveis no servidor: {available}")
        for name in (model_a, model_b):
            if name not in available:
                print(f"  ⚠  '{name}' NÃO está na lista do servidor."
                      f" O servidor pode ignorar esse nome e usar outro modelo.")
    except Exception:
        print("  (não foi possível listar os modelos do servidor)")

    # Compara os vetores de probe
    probe_text = "The quick brown fox jumps over the lazy dog — embedding probe 12345"
    vec_a = embed(client_a, model_a, probe_text)
    vec_b = embed(client_b, model_b, probe_text)
    sim = _cosine_sim(vec_a, vec_b)

    print(f"  Similaridade cosseno dos probes: {sim:.6f}")
    if sim > 0.9999:
        print(
            "\n  ❌ ATENÇÃO: os vetores são praticamente idênticos (sim ≈ 1).\n"
            "     O servidor está ignorando o parâmetro 'model' e retornando\n"
            "     o mesmo embedding para ambos os nomes.\n"
            "\n"
            "     Possíveis correções:\n"
            "       1. Carregue dois servidores em portas diferentes e use\n"
            "          --base-url-a http://host:PORT_A/v1\n"
            "          --base-url-b http://host:PORT_B/v1\n"
            "       2. Use um servidor que suporte múltiplos modelos (vLLM,\n"
            "          text-embeddings-inference, Ollama multi-model).\n"
            "       3. Passe o ID exato que aparece em 'Modelos disponíveis'\n"
            "          acima para --model-a e --model-b.\n"
        )
    elif sim > 0.98:
        print(
            "  ⚠  Similaridade muito alta — modelos podem ser variações\n"
            "     do mesmo base model (ex.: quantizações diferentes)."
        )
    elif len(vec_a) == len(vec_b):
        print(f"  ✓  Dimensão igual ({len(vec_a)}d), mas vetores distintos — OK.")
    else:
        print(f"  ✓  Dimensões diferentes ({len(vec_a)}d vs {len(vec_b)}d) — modelos distintos confirmados.")

# ── Query presets ──────────────────────────────────────────────────────────────
#
# Escolha com --query-preset <nome> ou passe um arquivo com --queries.
# "generic-dotnet" é o padrão: funciona com qualquer repositório .NET.

QUERY_PRESETS: dict[str, list[str]] = {
    # Padrões universais de .NET — funciona com qualquer repositório C#
    "generic-dotnet": [
        "injeção de dependência e registro de serviços",
        "tratamento de erros e exceções globais",
        "autenticação e autorização",
        "configuração de banco de dados",
        "publicação e consumo de eventos",
        "logging e monitoramento",
        "validação de dados de entrada",
        "mapeamento entre objetos e DTOs",
        "chamada a serviço externo ou HTTP client",
        "background job ou worker service",
        "testes unitários de handler ou serviço",
        "configuração de middleware no pipeline HTTP",
    ],

    # Same as generic-dotnet but in English — removes language mismatch when
    # the codebase uses English identifiers/comments
    "english-dotnet": [
        "dependency injection and service registration",
        "global error handling and exception middleware",
        "authentication and authorization",
        "database configuration and connection setup",
        "publishing and consuming events or messages",
        "logging and monitoring",
        "input data validation",
        "object mapping and DTOs",
        "external service call or HTTP client",
        "background job or worker service",
        "unit tests for handler or service",
        "HTTP pipeline middleware configuration",
    ],

    # Específico para o projeto CleanArchitecture (CQRS + MediatR)
    "clean-arch": [
        "como criar um novo caso de uso",
        "validação de entrada com FluentValidation",
        "repositório de TodoItem",
        "autenticação e autorização de usuário",
        "mapeamento com AutoMapper",
        "tratamento de erros e exceções",
        "injeção de dependência e registro de serviços",
        "configuração de banco de dados com Entity Framework",
        "como deletar um item",
        "evento de domínio disparado após criação",
    ],

    # Específico para realtimematches (workers, crawlers, integrações)
    "realtimematches": [
        "crawler worker que busca dados externos",
        "publicação de evento no barramento de mensagens",
        "integração com provedor de dados esportivos",
        "processamento de partida em tempo real",
        "persistência de resultado de jogo",
        "autenticação e autorização de usuário",
        "tratamento de erro em worker ou background service",
        "configuração de infraestrutura e dependências",
        "endpoint que retorna dados de partida",
        "monitoramento e APM da aplicação",
    ],
}

DEFAULT_PRESET = "generic-dotnet"
DEFAULT_QUERIES = QUERY_PRESETS[DEFAULT_PRESET]


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def ensure_collection(qdrant: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        print(f"  coleção '{name}' já existe – reutilizando.")
        return
    qdrant.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    for field, schema in [
        ("file_path",     qm.PayloadSchemaType.KEYWORD),
        ("kind",          qm.PayloadSchemaType.KEYWORD),
        ("symbol",        qm.PayloadSchemaType.KEYWORD),
        ("content_hash",  qm.PayloadSchemaType.KEYWORD),
    ]:
        qdrant.create_payload_index(collection_name=name, field_name=field, field_schema=schema)


def stable_id(content_hash: str) -> int:
    return int(content_hash[:16], 16)


def index_chunks(
    qdrant: QdrantClient,
    client,
    chunks: list[dict],
    collection: str,
    model: str,
    batch: int = 32,
) -> tuple[int, float]:
    """Retorna (dim, tempo_total_segundos)."""
    print(f"\n[{model}] Sondando dimensão...")
    probe = embed(client, model, "probe")
    dim = len(probe)
    print(f"  dim = {dim}")

    ensure_collection(qdrant, collection, dim)

    # hashes já indexados
    seen: set[str] = set()
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
                seen.add(h)
        if next_offset is None:
            break

    to_index = [c for c in chunks if c["content_hash"] not in seen]
    print(f"  {len(seen)} já indexados, {len(to_index)} novos")

    if not to_index:
        return dim, 0.0

    buffer: list[qm.PointStruct] = []
    t0 = time.perf_counter()
    for chunk in tqdm(to_index, desc=f"Embedding [{model}]"):
        try:
            vector = embed(client, model, chunk["code"])
        except Exception as exc:
            print(f"  warn: falha em {chunk['symbol']}: {exc}", file=sys.stderr)
            continue
        buffer.append(
            qm.PointStruct(
                id=stable_id(chunk["content_hash"]),
                vector=vector,
                payload=chunk,
            )
        )
        if len(buffer) >= batch:
            qdrant.upsert(collection_name=collection, points=buffer)
            buffer.clear()
    if buffer:
        qdrant.upsert(collection_name=collection, points=buffer)
    elapsed = time.perf_counter() - t0
    return dim, elapsed


# ── Métricas ───────────────────────────────────────────────────────────────────

def spearman_corr(rank_a: list[str], rank_b: list[str]) -> float:
    """Correlação de Spearman baseada nos símbolos presentes nos dois rankings.

    Usa apenas a interseção para evitar distorção por símbolos ausentes num
    dos lados (que causava valores fora de [-1, 1] na fórmula simplificada).
    Retorna nan se a interseção for menor que 2 elementos.
    """
    pos_a = {sym: i for i, sym in enumerate(rank_a)}
    pos_b = {sym: i for i, sym in enumerate(rank_b)}
    common = [s for s in rank_a if s in pos_b]
    n = len(common)
    if n < 2:
        return float("nan")

    d_sq_sum = sum((pos_a[s] - pos_b[s]) ** 2 for s in common)
    rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
    # clamp defensivo: fórmula simplificada pode sair de [-1,1] com empates
    return max(-1.0, min(1.0, rho))


def query_stats(scores: list[float]) -> dict:
    if not scores:
        return {"top1": 0, "mean": 0, "std": 0, "gap": 0}
    return {
        "top1": scores[0],
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "gap": scores[0] - scores[-1],
    }


# ── Saída formatada ────────────────────────────────────────────────────────────

COL_W = 52

def section(title: str) -> None:
    print(f"\n{'─' * (COL_W * 2 + 7)}")
    print(f"  {title}")
    print(f"{'─' * (COL_W * 2 + 7)}")


def row(label: str, val_a: str, val_b: str) -> None:
    print(f"  {label:<22}  {val_a:<{COL_W}}  {val_b:<{COL_W}}")


def print_results_row(rank: int, r_a, r_b) -> None:
    def fmt(r) -> str:
        if r is None:
            return "(sem resultado)"
        p = r.payload or {}
        return (
            f"score={r.score:.4f}  {p.get('kind','?'):8}  "
            f"{(p.get('symbol') or '')[:28]:<28}\n"
            f"  {'':22}  {(p.get('file_path') or '')[-40:]}"
        )
    pa = r_a.payload or {} if r_a else {}
    pb = r_b.payload or {} if r_b else {}
    sa = f"score={r_a.score:.4f}  {pa.get('kind','?'):8}  {(pa.get('symbol') or '')[:26]:<26}" if r_a else "(sem resultado)"
    sb = f"score={r_b.score:.4f}  {pb.get('kind','?'):8}  {(pb.get('symbol') or '')[:26]:<26}" if r_b else "(sem resultado)"
    fa = (pa.get('file_path') or '')[-40:] if r_a else ""
    fb = (pb.get('file_path') or '')[-40:] if r_b else ""
    print(f"  #{rank:<3}  {sa:<{COL_W}}  {sb:<{COL_W}}")
    print(f"  {'':5}  {fa:<{COL_W}}  {fb:<{COL_W}}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Compara dois modelos de embedding no Qdrant")
    ap.add_argument("jsonl", type=Path, help="Arquivo JSONL de chunks")
    ap.add_argument("--model-a",  default="text-embedding-bge-m3")
    ap.add_argument("--model-b",  default="qwen3-embedding")
    ap.add_argument("--col-a",    default="bench_bgem3")
    ap.add_argument("--col-b",    default="bench_qwen3")
    ap.add_argument("--base-url", default=None, help="URL base do servidor (igual para ambos)")
    ap.add_argument("--base-url-a", default=None, help="URL base exclusiva para o modelo A")
    ap.add_argument("--base-url-b", default=None, help="URL base exclusiva para o modelo B")
    ap.add_argument("--sample",   type=int, default=0,
                    help="Limita N chunks para indexação rápida (0 = todos)")
    ap.add_argument("--skip-indexing", action="store_true",
                    help="Pula a etapa de indexação (coleções já devem existir)")
    ap.add_argument("--queries",  type=Path, default=None,
                    help="Arquivo .txt com uma query por linha (sobrepõe --query-preset)")
    ap.add_argument("--query-preset", default=DEFAULT_PRESET,
                    choices=list(QUERY_PRESETS),
                    help=f"Preset de queries a usar (default: {DEFAULT_PRESET}). "
                         f"Disponíveis: {', '.join(QUERY_PRESETS)}")
    ap.add_argument("--k",        type=int, default=5)
    ap.add_argument("--batch",    type=int, default=32)
    args = ap.parse_args()

    # ── queries ────────────────────────────────────────────────────────────────
    if args.queries and args.queries.exists():
        queries = [l.strip() for l in args.queries.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Queries carregadas de arquivo: {args.queries} ({len(queries)} queries)")
    else:
        queries = QUERY_PRESETS[args.query_preset]
        print(f"Usando preset '{args.query_preset}' ({len(queries)} queries)")

    # ── clientes ───────────────────────────────────────────────────────────────
    url_a = args.base_url_a or args.base_url
    url_b = args.base_url_b or args.base_url
    client_a = make_client(url_a)
    client_b = make_client(url_b)
    qdrant   = QdrantClient(url=QDRANT_URL)

    verify_distinct_models(client_a, client_b, args.model_a, args.model_b)

    # ── indexação ──────────────────────────────────────────────────────────────
    dim_a = dim_b = None
    time_a = time_b = 0.0

    if not args.skip_indexing:
        if not args.jsonl.exists():
            print(f"error: {args.jsonl} não encontrado", file=sys.stderr)
            return 1

        chunks = [json.loads(l) for l in args.jsonl.open(encoding="utf-8")]
        if args.sample > 0:
            chunks = chunks[: args.sample]
        print(f"Chunks a indexar: {len(chunks)}")

        dim_a, time_a = index_chunks(qdrant, client_a, chunks, args.col_a, args.model_a, args.batch)
        dim_b, time_b = index_chunks(qdrant, client_b, chunks, args.col_b, args.model_b, args.batch)
    else:
        # sonda dimensões sem indexar
        probe_a = embed(client_a, args.model_a, "probe")
        probe_b = embed(client_b, args.model_b, "probe")
        dim_a, dim_b = len(probe_a), len(probe_b)

    # ── benchmark de queries ───────────────────────────────────────────────────
    section(f"COMPARAÇÃO: {args.model_a}  vs  {args.model_b}")
    row("Métrica", args.model_a, args.model_b)
    print(f"  {'─'*22}  {'─'*COL_W}  {'─'*COL_W}")
    if not args.skip_indexing:
        row("Tempo de indexação (s)", f"{time_a:.1f}", f"{time_b:.1f}")
    row("Dimensão (info)", str(dim_a), str(dim_b))

    # agregadores por modelo
    all_scores_a: list[float] = []
    all_scores_b: list[float] = []
    all_top1_a:   list[float] = []
    all_top1_b:   list[float] = []
    all_gaps_a:   list[float] = []
    all_gaps_b:   list[float] = []
    all_corr:     list[float] = []
    all_overlaps: list[float] = []
    agree_top1:   int = 0

    for qi, query in enumerate(queries, 1):
        section(f"Query #{qi}: «{query}»")
        row("Rank", args.model_a, args.model_b)
        print(f"  {'─'*5}  {'─'*COL_W}  {'─'*COL_W}")

        vec_a = embed(client_a, args.model_a, query)
        vec_b = embed(client_b, args.model_b, query)

        res_a = qdrant.search(collection_name=args.col_a, query_vector=vec_a,
                              limit=args.k, with_payload=True)
        res_b = qdrant.search(collection_name=args.col_b, query_vector=vec_b,
                              limit=args.k, with_payload=True)

        # side-by-side
        for i in range(args.k):
            ra = res_a[i] if i < len(res_a) else None
            rb = res_b[i] if i < len(res_b) else None
            print_results_row(i + 1, ra, rb)

        # estatísticas desta query
        scores_a = [r.score for r in res_a]
        scores_b = [r.score for r in res_b]
        st_a = query_stats(scores_a)
        st_b = query_stats(scores_b)

        all_scores_a.extend(scores_a)
        all_scores_b.extend(scores_b)
        all_top1_a.append(st_a["top1"])
        all_top1_b.append(st_b["top1"])
        all_gaps_a.append(st_a["gap"])
        all_gaps_b.append(st_b["gap"])

        syms_a = [(r.payload or {}).get("symbol", str(r.id)) for r in res_a]
        syms_b = [(r.payload or {}).get("symbol", str(r.id)) for r in res_b]
        corr = spearman_corr(syms_a, syms_b)
        all_corr.append(corr)

        top1_agree = bool(syms_a and syms_b and syms_a[0] == syms_b[0])
        if top1_agree:
            agree_top1 += 1

        # Normalized rank score: compara quais chunks do modelo A aparecem
        # no top-k do modelo B (overlap@k) — não depende de escala de scores
        overlap = len(set(syms_a) & set(syms_b))
        overlap_pct = overlap / args.k if args.k > 0 else 0.0
        all_overlaps.append(overlap_pct)

        print()
        row("score@1 (resultado mais relevante)",
            f"{st_a['top1']:.4f}",
            f"{st_b['top1']:.4f}")
        row("score médio (top-k)",
            f"{st_a['mean']:.4f} ± {st_a['std']:.4f}",
            f"{st_b['mean']:.4f} ± {st_b['std']:.4f}")
        row("gap rank1−rankK",
            f"{st_a['gap']:.4f}",
            f"{st_b['gap']:.4f}")
        row("top-1 concorda?", "sim" if top1_agree else "não", "")
        row(f"overlap@{args.k} (chunks em comum)", f"{overlap}/{args.k} ({overlap_pct:.0%})", "")
        corr_str = f"{corr:.3f}" if corr == corr else "n/a (sem interseção)"
        row("rank corr (Spearman, interseção)", corr_str, "")

    # ── resumo global ──────────────────────────────────────────────────────────
    section("RESUMO GLOBAL")
    row("Métrica", args.model_a, args.model_b)
    print(f"  {'─'*22}  {'─'*COL_W}  {'─'*COL_W}")

    def g_mean(lst): return statistics.mean(lst) if lst else float("nan")
    def g_std(lst):  return statistics.stdev(lst) if len(lst) > 1 else 0.0

    n_queries = len(queries)
    row("Score@1 médio",
        f"{g_mean(all_top1_a):.4f}",
        f"{g_mean(all_top1_b):.4f}")
    row("Score médio (top-k completo)",
        f"{g_mean(all_scores_a):.4f}",
        f"{g_mean(all_scores_b):.4f}")
    row("Score std (top-k)",
        f"{g_std(all_scores_a):.4f}",
        f"{g_std(all_scores_b):.4f}")
    row("Gap médio rank1−rankK",
        f"{g_mean(all_gaps_a):.4f}",
        f"{g_mean(all_gaps_b):.4f}")
    row("Concordância top-1",
        f"{agree_top1}/{n_queries} queries ({100*agree_top1/n_queries:.0f}%)", "")
    row(f"Overlap@{args.k} médio (chunks em comum)",
        f"{g_mean(all_overlaps):.0%}", "")
    valid_corr = [c for c in all_corr if c == c]  # remove nan
    corr_str = f"{g_mean(valid_corr):.3f}" if valid_corr else "n/a"
    row("Rank corr média (Spearman, interseção)", corr_str, "")
    print()
    print("  Nota: scores absolutos não são comparáveis entre modelos com")
    print("  espaços vetoriais diferentes (bge-m3=1024d, qwen3=2560d).")
    print("  Use overlap@k e concordância top-1 como métricas model-agnósticas.")

    # ── interpretação automática ───────────────────────────────────────────────
    section("INTERPRETAÇÃO")
    valid_corr_list = [c for c in all_corr if c == c]
    corr_avg = g_mean(valid_corr_list) if valid_corr_list else float("nan")

    # pontuação de qualidade composta: score@1 (peso 3) + gap (peso 2) + score_mean (peso 1)
    # normalizados pelo maior valor entre os dois modelos
    def norm(a, b): return (a / b if b else 1.0) if b >= a else (b / a if a else 1.0)
    s1a, s1b = g_mean(all_top1_a), g_mean(all_top1_b)
    ga,  gb  = g_mean(all_gaps_a),  g_mean(all_gaps_b)
    ma,  mb  = g_mean(all_scores_a), g_mean(all_scores_b)

    pts_a = (3 * s1a + 2 * ga + 1 * ma)
    pts_b = (3 * s1b + 2 * gb + 1 * mb)
    winner = args.model_a if pts_a >= pts_b else args.model_b
    loser  = args.model_b if pts_a >= pts_b else args.model_a
    margin = abs(pts_a - pts_b) / max(pts_a, pts_b) * 100

    winner_top1 = args.model_a if s1a >= s1b else args.model_b
    winner_gap  = args.model_a if ga  >= gb  else args.model_b

    print(f"  ★  VENCEDOR EM QUALIDADE: {winner}  (margem: {margin:.1f}%)")
    print()
    print(f"  Score@1 maior  →  {winner_top1}  ({s1a:.4f} vs {s1b:.4f})")
    print(f"    score@1 é o indicador mais crítico para RAG: mede o quão")
    print(f"    próxima semanticamente o modelo coloca a query do chunk #1.")
    print()
    print(f"  Gap rank1−rankK maior  →  {winner_gap}  ({ga:.4f} vs {gb:.4f})")
    print(f"    gap alto = modelo separa bem o chunk relevante dos irrelevantes,")
    print(f"    reduzindo risco de ruído no contexto enviado ao LLM.")
    print()
    print(f"  Concordância top-1 entre modelos: {agree_top1}/{n_queries} queries ({100*agree_top1/n_queries:.0f}%)")
    print(f"  Correlação Spearman média: {corr_avg:.3f}")
    if corr_avg > 0.7:
        print("    → Modelos concordam sobre o que é relevante (rankings similares)")
    elif corr_avg > 0.4:
        print("    → Concordância moderada: modelos capturam nuances diferentes do código")
    else:
        print("    → Baixa concordância: modelos têm visões semânticas bem distintas —")
        print("      vale revisar manualmente qual ranking faz mais sentido para seu domínio")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
