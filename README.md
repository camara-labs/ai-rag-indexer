# RAG Indexer — Multi-Language Code Search Pipeline

A semantic chunking and indexing pipeline that turns source code repositories into a searchable vector database. Query your codebase in natural language and get precise, cited answers from an LLM.

```
[1] Chunking  →  [2] Summarization (optional)  →  [3] Embedding  →  [4] Qdrant Storage  →  [5] RAG Query
```

Keeping the stages separate means you can swap the embedding model without re-parsing, inspect chunks before indexing, and version the JSONL output.

---

## Supported Languages

| Language | Extensions | Chunk types |
|---|---|---|
| **C#** | `.cs` | method, constructor, property, class |
| **TypeScript** | `.ts`, `.tsx` | function, class, method, constructor, interface, type |
| **JavaScript** | `.js`, `.jsx` | function, class, method, constructor |
| **Terraform** | `.tf` | resource, data, module, variable, output, locals, provider |

---

## Why Semantic Chunking

Splitting code every N lines breaks methods in half, loses class context, and mixes imports with logic. The resulting embedding is generic and retrieval returns noise.

This pipeline uses **Tree-sitter** to parse source code into an AST and extract complete semantic units:

- Each **method, function, constructor, and property** becomes its own chunk.
- **Small classes** (≤ 60 lines — DTOs, records, interfaces, enums) are kept as a single chunk.
- **Large classes** are broken down member by member.
- Each chunk carries a **context header** with `// File:`, `// Namespace:`, `// Class:`, and the file-level imports — this is what lets the embedding distinguish `Save()` in `OrderRepository` from `Save()` in `UserRepository`.
- **Leading doc comments** (XML `///`, JSDoc `/** */`) immediately above a member are included in its chunk.
- Terraform chunks carry the full block (`resource "aws_s3_bucket" "my_bucket" { ... }`) as a single unit.

---

## Requirements

- Python 3.12+
- Qdrant (via Docker)
- An OpenAI-compatible embedding server (Ollama, LM Studio, vLLM, llama.cpp)
- An OpenAI-compatible LLM server (same options)

---

## Installation

```bash
cd indexer

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows (bash)
# or: .venv\Scripts\Activate.ps1    # Windows (PowerShell)
# or: source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
OPENAI_BASE_URL=http://localhost:1234/v1   # embedding server
OPENAI_API_KEY=lm-studio                   # any string for local servers
EMBED_MODEL=text-embedding-bge-m3

LLM_BASE_URL=http://localhost:1234/v1      # chat/LLM server (can be the same)
LLM_MODEL=qwen3-4b

QDRANT_URL=http://localhost:6333
```

Start Qdrant:

```bash
docker compose up -d
```

---

## Full Pipeline — Interactive CLI

The easiest way to run the full pipeline (chunk → embed → store):

```bash
python cli.py
```

Languages are **auto-detected** by scanning the repository for known file extensions — no language prompt. A repository with TypeScript and Terraform files will index both automatically.

You will be prompted for:
1. **Repository path** — absolute path to the directory to index
2. **Collection name** — Qdrant collection name
3. **JSONL output path** — intermediate chunk file (defaults to `chunks/<collection>.jsonl`)
4. **Embedding model** — defaults to `EMBED_MODEL` from `.env`
5. **Summarization** — whether to run an LLM summarization step before embedding (default: No)
   - If yes, the LLM model is also prompted (defaults to `LLM_MODEL` from `.env`)

The confirmation screen shows the full configuration before proceeding:

```
  Languages  : typescript, terraform (auto-detected)
  Repo path  : /home/user/my-infra-app
  Collection : my_infra_app
  JSONL path : chunks/my_infra_app.jsonl
  Embed model: (from .env)
  Max chunk  : (no limit)
  Summarize  : yes (model=qwen3-4b)
Proceed with indexing? [Y/n]
```

All prompts can be bypassed with flags for scripted use:

```bash
python cli.py \
  --repo-path /abs/path/to/my-app \
  --collection my_app \
  --yes
```

Enable summarization non-interactively with `--summarize`:

```bash
python cli.py \
  --repo-path /abs/path/to/my-app \
  --collection my_app \
  --summarize \
  --llm-model qwen3-4b \
  --yes
```

Use `--no-summarize` to skip the prompt explicitly:

```bash
python cli.py \
  --repo-path /abs/path/to/my-app \
  --collection my_app \
  --no-summarize \
  --yes
```

Use `--language` to restrict indexing to a single language when needed:

```bash
python cli.py \
  --repo-path /abs/path/to/my-app \
  --collection my_app_ts \
  --language typescript \
  --yes
```

---

## Individual Pipeline Stages

### Stage 1 — Chunking

Parse a repository and write a JSONL of semantic chunks. Languages are auto-detected:

```bash
python chunker.py /path/to/repo chunks/my-app.jsonl
```

Restrict to a single language when needed:

```bash
python chunker.py /path/to/repo chunks/my-app.jsonl --language typescript
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--language` | auto-detect | Restrict to one language |
| `--max-chunk-chars N` | no limit | Skip chunks larger than N characters |
| `--top-large N` | 5 | Print the N largest chunks (inspection) |

Automatically skipped paths per language:

| Language | Skipped directories | Skipped files |
|---|---|---|
| C# | `bin/`, `obj/`, `Migrations/` | `*.Designer.cs`, `*.g.cs`, `AssemblyInfo.cs` |
| TypeScript | `node_modules/`, `dist/`, `.next/`, `coverage/` | `*.d.ts` |
| JavaScript | `node_modules/`, `dist/`, `.next/`, `coverage/` | `*.min.js` |
| Terraform | `.terraform/` | `*.tfstate`, `*.tfstate.backup` |

### Stage 2 — Embedding

Embed the chunks and add a `"vector"` field to each:

```bash
python embedder.py chunks/my-app.jsonl \
  --model text-embedding-bge-m3 \
  --output chunks/my-app-embedded.jsonl
```

### Stage 3 — Storage

Upsert embedded chunks into Qdrant (incremental — skips already-indexed hashes):

```bash
python storer.py chunks/my-app-embedded.jsonl --collection my_app_ts
```

---

## Querying

Ask a question in natural language. The system embeds the question, retrieves the top-K most relevant chunks, and streams an LLM answer:

```bash
python ask.py "how does authentication work?" --collection my_app_ts
```

The language is inferred from the retrieved chunks' `language` field — the appropriate system prompt is selected automatically. You can also force a language:

```bash
python ask.py "which resources create S3 buckets?" \
  --collection infra_tf \
  --language terraform
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--collection` | prompted | Qdrant collection to query |
| `--k N` | 5 | Number of chunks to retrieve |
| `--language` | auto (from chunks) | Force a specific system prompt |
| `--embed-model` | `EMBED_MODEL` env | Embedding model |
| `--llm-model` | `LLM_MODEL` env | Chat model |
| `--show-context` | off | Print retrieved chunks before the answer |
| `--max-ctx N` | 3500 | Token budget for the prompt |
| `--temperature` | 0.2 | LLM temperature |

Each language has a dedicated system prompt that gives the LLM the right domain context (C#/.NET, TypeScript/Node, JavaScript, Terraform/IaC).

---

## Summarization (Optional)

The summarizer adds a `summary` field to each chunk with a 1–3 sentence LLM description of what the code does. Summaries are stored as payload in Qdrant alongside the chunk, enriching retrieval context.

### Via the CLI (recommended)

The summarization step is offered interactively during `python cli.py`. It runs between chunking and embedding, and the JSONL is updated in place with the summaries before they are indexed.

For scripted use:

```bash
python cli.py \
  --repo-path /abs/path/to/my-app \
  --collection my_app \
  --summarize \
  --llm-model qwen3-4b \
  --summary-max-tokens 256 \
  --yes
```

CLI options for summarization:

| Flag | Default | Description |
|---|---|---|
| `--summarize` / `--no-summarize` | ask interactively | Enable or skip the summarization step |
| `--llm-model` | `LLM_MODEL` env | LLM model id for summarization |
| `--llm-base-url` | `LLM_BASE_URL` env | LLM server base URL |
| `--summary-max-tokens N` | 256 | Max tokens per summary response |

### Standalone

Run the summarizer on an existing JSONL independently:

```bash
python summarizer.py chunks/my-app.jsonl --model qwen3-4b
```

Writes to `chunks/my-app-summarized.jsonl`. Supports checkpoint resumption — safe to interrupt and restart.

| Flag | Default | Description |
|---|---|---|
| `--output` | `<input>-summarized.jsonl` | Output path |
| `--model` | `LLM_MODEL` env | Chat model to use |
| `--max-tokens N` | 4096 | Max tokens per summary |
| `--skip-existing` | off | Skip chunks that already have a `summary` field |

---

## JSONL Schema

Each line is a JSON object:

```json
{
  "code":         "// File: src/auth/AuthService.ts\n// Class: AuthService\nimport { Injectable } ...\n\nasync findById(id: string): Promise<User> { ... }",
  "file_path":    "src/auth/AuthService.ts",
  "namespace":    "auth",
  "class_name":   "AuthService",
  "symbol":       "AuthService.findById",
  "kind":         "method",
  "signature":    "async findById(id: string): Promise<User>",
  "imports":      ["import { Injectable } from '@angular/core';"],
  "start_line":   24,
  "end_line":     30,
  "content_hash": "a3f1c2d4...",
  "language":     "typescript",
  "summary":      "Retrieves a User by ID from the database, returning null if not found."
}
```

Key fields:

| Field | Purpose |
|---|---|
| `code` | Text that becomes the embedding and goes into the LLM prompt |
| `language` | Source language — used for system prompt selection and code block syntax highlighting |
| `symbol` | Fully-qualified member name — useful for deduplication and display |
| `kind` | `method` / `function` / `constructor` / `property` / `class` / `interface` / `type` / `resource` / `variable` / ... |
| `file_path`, `start_line`, `end_line` | "Open in editor" navigation and neighbour-context expansion |
| `content_hash` | Incremental indexing — only re-embeds changed chunks |
| `summary` | *(optional)* LLM-generated 1–3 sentence description — added by the summarization step |

---

## Inspecting Chunks

Count by type:

```bash
python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('chunks/my-app.jsonl', encoding='utf-8')]
print(Counter((r['language'], r['kind']) for r in rows))
"
```

Print the first chunk formatted:

```bash
python -c "
import json
print(json.dumps(json.loads(open('chunks/my-app.jsonl', encoding='utf-8').readline()), indent=2))
"
```

Inspect the AST of a single file (useful when tuning a chunker):

```bash
python inspect_ast.py path/to/file.cs
```

**Always review a sample of chunks before embedding.** If an important method was split wrong or a block is missing context, this is where you find it — cheaply.

---

## Project Structure

```
indexer/
├── cli.py              # Interactive full-pipeline CLI
├── pipeline.py         # Programmatic pipeline orchestrator
├── chunker.py          # Standalone chunking CLI
├── embedder.py         # Stage 2: add vectors to chunks
├── storer.py           # Stage 3: upsert to Qdrant
├── ask.py              # RAG query with streaming LLM answer
├── summarizer.py       # Optional: add LLM summaries to chunks
├── chat.py             # Multi-turn conversational RAG
├── llm_client.py       # Shared OpenAI-compatible client factory
├── chunkers/
│   ├── __init__.py     # Language dispatcher
│   ├── base.py         # Chunk dataclass + shared AST helpers
│   ├── csharp.py       # C# chunker (tree-sitter-c-sharp)
│   ├── typescript.py   # TypeScript/TSX chunker (tree-sitter-typescript)
│   ├── javascript.py   # JavaScript/JSX chunker (tree-sitter-javascript)
│   └── terraform.py    # Terraform/HCL chunker (tree-sitter-hcl)
├── chunks/             # Intermediate JSONL files
├── sessions/           # Chat session histories
├── plans/              # Implementation planning documents
└── requirements.txt
```

---

## Adding a New Language

1. **Create `chunkers/<language>.py`**
   - Implement `chunk_file(path, repo_root) -> list[Chunk]` using the appropriate `tree-sitter-<language>` parser
   - Implement `iter_<ext>_files(root)` with the right skip rules
   - Implement `chunk_repo(repo_path) -> list[Chunk]`
   - Pass `language="<language>"` to every `_make_chunk()` call

2. **Register in `chunkers/__init__.py`**
   - Add to `_SUPPORTED` tuple
   - Add an `if lang == "<language>"` branch

3. **Add to `cli.py`**
   - Add to `_SUPPORTED_LANGUAGES`

4. **Add a system prompt to `ask.py`**
   - Add an entry to `_SYSTEM_PROMPTS` with domain-appropriate instructions

5. **Add the parser package to `requirements.txt`**

---

## Embedding Model Recommendations

| Model | Dims | Notes |
|---|---|---|
| `text-embedding-bge-m3` | 1024 | **Recommended.** Multilingual (PT-BR + EN), code-aware, supports hybrid search. |
| `nomic-embed-text` | 768 | English-only, poor on code. Retrieval quality degrades significantly for non-English queries. |

**Important:** the embedding model is a contract with the Qdrant collection. A collection created with 1024-dim vectors cannot accept 768-dim vectors. When switching models, create a new collection — do not reindex in place.

Naming convention used in this project: `<codebase>_<model>`, e.g. `clean_arch_bgem3`. This makes the model explicit in every query command and allows keeping old collections for side-by-side comparison.

---

## Known Limitations

- **Very large methods (> ~800 tokens)** are not sub-split. If you have many, add a token-aware splitter before embedding.
- **Type resolution** is not performed — only syntactic structure. Semantic cross-references (e.g. "all callers of this method") require a full language server.
- **Top-level statements** in C# (programs without a class) and bare scripts in JS/TS are not extracted as named chunks.
- **Terraform `locals` blocks** produce a single chunk per `locals {}` block; individual local values are not split out.
