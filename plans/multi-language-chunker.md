# Planejamento: Multi-Language Chunker

## Objetivo

Evoluir o indexer de mono-linguagem (C#) para suportar múltiplas linguagens:
**C#** (já implementado), **TypeScript**, **JavaScript** e **Terraform**.

---

## Fase 1 — Dependências

Adicionar os pacotes tree-sitter em `requirements.txt`:

| Linguagem  | Pacote PyPI               |
|------------|---------------------------|
| TypeScript | `tree-sitter-typescript`  |
| JavaScript | `tree-sitter-javascript`  |
| Terraform  | `tree-sitter-hcl`         |

---

## Fase 2 — Base: Campo `language` no Chunk Dataclass

**`chunkers/base.py`** — adicionar o campo `language` ao dataclass `Chunk`:

```python
@dataclass
class Chunk:
    ...
    language: str  # "csharp" | "typescript" | "javascript" | "terraform"
```

- Cada chunker preenche o campo ao criar chunks via `_make_chunk()`
- Elimina a necessidade de inferir linguagem por extensão de arquivo nos scripts downstream
- Atualizar `csharp.py` para preencher `language="csharp"`

---

## Fase 3 — Novos Chunkers

Criar três novos arquivos em `chunkers/`:

### `chunkers/typescript.py`

- Parser: `tree-sitter-typescript` (suporta `.ts` e `.tsx`)
- Nós relevantes: `function_declaration`, `class_declaration`, `method_definition`, `arrow_function` exportado, `interface_declaration`, `type_alias_declaration`
- Filtros: pular `node_modules/`, `dist/`, arquivos `.d.ts` e gerados
- Convenção de símbolo: `ClassName.methodName` (igual ao C#)
- Campo `language="typescript"`

### `chunkers/javascript.py`

- Parser: `tree-sitter-javascript` (suporta `.js` e `.jsx`)
- Lógica similar ao TypeScript; sem interfaces e type aliases
- Filtros: pular `node_modules/`, `dist/`, `*.min.js`, arquivos bundle
- Campo `language="javascript"`

### `chunkers/terraform.py`

- Parser: `tree-sitter-hcl`
- Nós relevantes: `resource`, `data`, `module`, `variable`, `output`, `locals`
- Chunks: um por bloco `resource`/`data`/`module`; variáveis e outputs podem ser agrupados por arquivo
- Convenção de símbolo: `resource.aws_s3_bucket.my_bucket`, `module.networking`, etc.
- `kind`: `"resource"`, `"data"`, `"module"`, `"variable"`, `"output"`
- Filtros: pular `.terraform/`, `*.tfstate`, `*.tfstate.backup`
- Campo `language="terraform"`

---

## Fase 4 — Dispatcher e CLI

### `chunkers/__init__.py`

```python
# De:
_SUPPORTED = ("csharp",)

# Para:
_SUPPORTED = ("csharp", "typescript", "javascript", "terraform")

# Dispatcher:
elif lang == "typescript": from . import typescript; return typescript.chunk_repo(...)
elif lang == "javascript": from . import javascript; return javascript.chunk_repo(...)
elif lang == "terraform":  from . import terraform;  return terraform.chunk_repo(...)
```

### `cli.py`

Única mudança necessária:

```python
# De:
_SUPPORTED_LANGUAGES = ["csharp"]

# Para:
_SUPPORTED_LANGUAGES = ["csharp", "javascript", "terraform", "typescript"]
```

O fluxo de prompts interativo já é genérico — nenhuma outra mudança.

---

## Fase 5 — Summarizer

**`summarizer.py`** — o bloco de código está hardcoded como ` ```csharp `.

- Usar o campo `language` do chunk (definido na Fase 2) para tornar o bloco dinâmico
- Substituir ` ```csharp ` por ` ```{chunk['language']} `

Exemplo:

```python
# De:
code_block = f"```csharp\n{chunk['code']}\n```"

# Para:
lang = chunk.get("language", "csharp")
code_block = f"```{lang}\n{chunk['code']}\n```"
```

---

## Fase 6 — Ask / RAG

**`ask.py`** — system prompt está em português assumindo C# e usa ` ```csharp ` no contexto.

Usar um **prompt por linguagem** para ser mais assertivo na geração de respostas:

```python
_SYSTEM_PROMPTS = {
    "csharp": (
        "Você é um assistente especialista em C# e .NET. "
        "Responda com base nos trechos de código fornecidos. "
        "Seja preciso, técnico e direto."
    ),
    "typescript": (
        "Você é um assistente especialista em TypeScript e Node.js/browser. "
        "Responda com base nos trechos de código fornecidos. "
        "Seja preciso, técnico e direto."
    ),
    "javascript": (
        "Você é um assistente especialista em JavaScript. "
        "Responda com base nos trechos de código fornecidos. "
        "Seja preciso, técnico e direto."
    ),
    "terraform": (
        "Você é um assistente especialista em Terraform e infraestrutura como código. "
        "Responda com base nos trechos de configuração HCL fornecidos. "
        "Seja preciso, técnico e direto."
    ),
}
```

- A linguagem pode ser inferida dos chunks retornados (campo `language`) ou passada via `--language`
- O bloco de código no contexto também usa o campo `language` do chunk

---

## Resumo de Arquivos Impactados

| Arquivo                     | Tipo de Mudança                             | Esforço  |
|-----------------------------|---------------------------------------------|----------|
| `requirements.txt`          | +3 pacotes tree-sitter                      | Trivial  |
| `chunkers/base.py`          | +campo `language` no `Chunk` dataclass      | Pequeno  |
| `chunkers/csharp.py`        | Preencher `language="csharp"` nos chunks    | Trivial  |
| `chunkers/__init__.py`      | +3 rotas no dispatcher                      | Pequeno  |
| `chunkers/typescript.py`    | Novo (inspirado em `csharp.py`)             | Médio    |
| `chunkers/javascript.py`    | Novo (similar ao TypeScript)                | Médio    |
| `chunkers/terraform.py`     | Novo (lógica distinta para HCL)             | Médio    |
| `cli.py`                    | +3 linguagens na lista `_SUPPORTED_LANGUAGES` | Trivial |
| `summarizer.py`             | Código block dinâmico via `chunk['language']` | Pequeno |
| `ask.py`                    | Prompt por linguagem + code block dinâmico  | Pequeno  |

---

## Ordem de Implementação Sugerida

1. **`base.py`** — adicionar campo `language` no `Chunk` + atualizar `csharp.py`
2. **`requirements.txt`** — instalar novas dependências
3. **`chunkers/typescript.py`** + **`chunkers/javascript.py`** — maior similaridade facilita fazer juntos
4. **`chunkers/terraform.py`** — lógica própria, testar isolado
5. **`chunkers/__init__.py`** + **`cli.py`** — registrar tudo
6. **`summarizer.py`** + **`ask.py`** — cleanup dos hardcodes e prompts por linguagem
