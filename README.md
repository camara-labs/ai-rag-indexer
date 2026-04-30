# Indexer — Etapa 1: Chunking semântico de C#

Este módulo lê um diretório com código C# e produz um arquivo **JSONL** onde cada linha é um "pedaço" (chunk) coerente do código — um método, propriedade, construtor ou classe pequena — com metadados estruturais.

Esta é a **primeira de três etapas** do pipeline de RAG:

```
[1] Chunking      ← VOCÊ ESTÁ AQUI   produz chunks.jsonl
[2] Embedding         (próxima etapa) produz vetores
[3] Indexação Qdrant  (próxima etapa) faz upsert no banco
```

Manter as etapas separadas permite trocar o modelo de embedding sem reparsear o código, inspecionar os chunks antes de indexar, e versionar o JSONL.

---

## Por que chunking semântico (e não por linhas)

Cortar código a cada N linhas quebra métodos no meio, perde o nome da classe e mistura `using`s com lógica. O resultado é que o embedding fica genérico e o retrieval traz lixo.

Aqui usamos **Tree-sitter** para parsear o C# em uma árvore sintática (AST) e extrair unidades inteiras:

- Cada **método**, **construtor** e **propriedade** vira um chunk.
- **Classes pequenas** (≤60 linhas — DTOs, records, enums) ficam inteiras em um único chunk.
- **Classes grandes** são quebradas membro a membro.
- Cada chunk leva no topo um **header de contexto** com `// File:`, `// Namespace:`, `// Class:` e os `using`s do arquivo — isso é o que permite ao embedding distinguir `Save()` de `OrderRepository` do `Save()` de `UserRepository`.
- **Comentários XML doc** (`/// <summary>`) imediatamente acima de um membro são incluídos no chunk dele.

---

## Pré-requisitos

- Python 3.12+ (incluindo 3.14)
- Pip

Não precisa compilar nada — `tree-sitter-c-sharp` já vem com o parser pré-buildado.

---

## Instalação

A partir da raiz do projeto:

```bash
cd indexer
C:/python/python.exe -m venv .venv
# Windows (bash):
source .venv/Scripts/activate
# (ou no PowerShell: .venv\Scripts\Activate.ps1)

pip install -r requirements.txt
```

Se o `py -3.12` falhar no Windows com erro de caminho inexistente, use o executavel Python que realmente existe no sistema (ex.: `C:/python/python.exe`):

```bash
C:/python/python.exe -m venv .venv
```

## Erro comum no Windows (`tree-sitter`)

Se aparecer `Building wheel for tree-sitter ... Microsoft Visual C++ 14.0 or greater is required`, normalmente voce esta com versoes antigas de dependencias em `requirements.txt` (sem wheel compativel) ou um ambiente inconsistente.

Resolucao recomendada:

```bash
deactivate
Remove-Item -Recurse -Force .venv
C:/python/python.exe -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Alternativa: instalar o Build Tools do Visual C++, mas para este projeto o ideal e usar as versoes atuais do `requirements.txt`.

---

## Uso

```bash
python chunker.py <diretório_código> <arquivo_saída.jsonl>
```

Exemplo, indexando o `CleanArchitecture` que já está no repo:

```bash
python chunker.py ../codebase/CleanArchitecture chunks/clean-arch.jsonl
```

Saída esperada no console:

```
Parsed 142 files, produced 873 chunks -> chunks/clean-arch.jsonl
```

Pastas `bin/`, `obj/`, `.git/`, `node_modules/` e `packages/` são ignoradas automaticamente.

---

## Formato do JSONL

Cada linha é um JSON com a estrutura:

```json
{
  "code": "// File: src/Auth/AuthService.cs\n// Namespace: MyApp.Auth\n// Class: AuthService\nusing System;\nusing Microsoft.IdentityModel.Tokens;\n\n/// <summary>Valida JWT...</summary>\npublic bool ValidateToken(string jwt) { ... }",
  "file_path": "src/Auth/AuthService.cs",
  "namespace": "MyApp.Auth",
  "class_name": "AuthService",
  "symbol": "AuthService.ValidateToken",
  "kind": "method",
  "signature": "public bool ValidateToken(string jwt)",
  "imports": ["using System;", "using Microsoft.IdentityModel.Tokens;"],
  "start_line": 42,
  "end_line": 58,
  "content_hash": "a3f1c2..."
}
```

Campos importantes:

| Campo          | Para que serve depois |
|----------------|---|
| `code`         | Texto que vai virar embedding e ir para o prompt do LLM |
| `symbol`       | Identificador único do membro — útil para deduplicar e exibir |
| `kind`         | `method` / `constructor` / `property` / `class` — útil para filtros no Qdrant |
| `file_path`, `start_line`, `end_line` | "Abrir no editor" e expandir contexto com vizinhos |
| `content_hash` | Indexação incremental — só re-embeda chunks alterados |

---

## Inspecionando o resultado

Conte chunks por tipo:

```bash
python -c "import json; from collections import Counter; print(Counter(json.loads(l)['kind'] for l in open('chunks/clean-arch.jsonl', encoding='utf-8')))"
```

Veja o primeiro chunk formatado:

```bash
python -c "import json; print(json.dumps(json.loads(open('chunks/clean-arch.jsonl', encoding='utf-8').readline()), indent=2, ensure_ascii=False))"
```

Olhe alguns chunks à mão antes de seguir para o embedding. Se algum método importante foi cortado errado, é aqui que você descobre — e barato.

---

## Limitações conhecidas (e quando se importar)

- **Métodos enormes (>800 tokens)** não são sub-divididos. Se você tiver muitos, adicionamos um splitter na etapa 2 antes de embedar.
- **Não resolve referências de tipos.** Para isso seria necessário Roslyn (análise semântica). Por ora, a expansão de contexto será feita por proximidade no arquivo, no momento do retrieval.
- **Top-level statements** (programas sem classe) não são extraídos como chunk separado — caem como filhos do root e são ignorados. Aceitável para a maioria dos projetos.

---

---

# Etapa 2 — Embeddings + Qdrant

## Pré-requisitos

1. **Qdrant rodando** (a partir da raiz do repo):

   ```bash
   docker compose up -d
   ```

   Confirme acessando http://localhost:6333/dashboard.

2. **Modelo de embedding no Ollama** — use `bge-m3`:

   ```bash
   ollama pull bge-m3
   ```

   1024 dimensões, multilíngue (PT-BR + EN), 567M parâmetros, roda rápido em CPU. **Não use `nomic-embed-text`** para esse projeto — ver "Lições aprendidas" abaixo.

3. As novas dependências Python já estão em `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Indexar os chunks

```bash
python index_chunks.py chunks/clean-arch.jsonl --model bge-m3 --collection csharp_code_bgem3
```

O que acontece:

1. Detecta automaticamente a dimensão do modelo (sondagem inicial).
2. Cria a coleção `csharp_code` no Qdrant se não existir, com índices de payload em `file_path`, `namespace`, `kind`, `symbol`, `content_hash`.
3. Lê todos os `content_hash` já indexados (se existirem) — **indexação incremental**, só re-embeda chunks novos ou alterados.
4. Para cada chunk, chama o Ollama para gerar o embedding e faz upsert no Qdrant com o JSON inteiro como payload.

Saída esperada:

```
Sondando dimensão do modelo bge-m3...
  dim = 1024
Lendo hashes já indexados...
  0 chunks já no Qdrant
Total: 157 chunks no JSONL, 157 novos para indexar
Embedding: 100%|██████████| 157/157 [00:19<00:00,  7.98it/s]
Concluído. Coleção 'csharp_code_bgem3' agora tem 157 pontos.
```

Reexecutar o comando depois deve dizer "Nada novo" — isso prova que o incremental funciona.

## Buscar

```bash
python search.py "como deletar um TodoItem do banco?" --collection csharp_code_bgem3 --model bge-m3
```

> ⚠️ **Importante**: o `--model` da busca **precisa ser o mesmo** usado na indexação. Vetores de modelos diferentes não são comparáveis (e dimensões diferentes nem sequer cabem na mesma coleção). Hoje isso é manual; uma melhoria futura é gravar o modelo nos metadados da coleção.

Saída esperada (top 5 por similaridade de cosseno):

```
#1  score=0.6283  class  DeleteTodoItemCommandHandler
    src/Application/TodoItems/Commands/DeleteTodoItem/DeleteTodoItem.cs:7-28
    namespace: CleanArchitecture.Application.TodoItems.Commands.DeleteTodoItem
    signature: public class DeleteTodoItemCommandHandler : IRequestHandler<DeleteTodoItemCommand>

#2  score=0.6168  class  CreateTodoItemCommandHandler
    ...
```

Filtrar por tipo (só métodos):

```bash
python search.py "validar token JWT" --collection csharp_code_bgem3 --model bge-m3 --kind method
```

---

## Lições aprendidas (ordem importa)

Esta seção registra os erros reais que cometemos ao montar a Etapa 2 e por que as escolhas atuais são as escolhas atuais. Leia antes de "otimizar" qualquer coisa aqui.

### 1. `nomic-embed-text` é ruim para código

Foi nosso primeiro modelo (porque é o default em vários tutoriais de RAG). Resultado das três queries de teste com a coleção indexada por ele:

| Query | DeleteTodoItemCommandHandler aparece? |
|---|---|
| `"como deletar um TodoItem do banco?"` (PT-BR) | ❌ não no top 5 — top 1 era `ForbiddenAccessException` |
| `"delete todo item from database"` (EN) | ❌ não no top 5 — top 1 era `TodoItemCompletedEvent` |
| `"Remove SaveChangesAsync TodoItems"` (lexical) | ❌ idem |

**Por que falhou:**

- `nomic-embed-text` é treinado em **texto natural em inglês**. Ele "vê" `_context.TodoItems.Remove(entity)` como string opaca, sem entender que isso é "deletar item".
- É **monolíngue** (inglês). Queries em PT-BR ficam num espaço vetorial diferente do código.

### 2. `bge-m3` resolveu os dois problemas

Mesma indexação, mesmo retrieval, mesmas três queries:

| Query | Top 1 |
|---|---|
| `"como deletar um TodoItem do banco?"` | ✅ `DeleteTodoItemCommandHandler` (score 0.628) |
| `"delete todo item from database"` | ✅ `DeleteTodoItemCommandHandler` (score 0.635) |
| `"Remove SaveChangesAsync TodoItems"` | ✅ `DeleteTodoItemCommandHandler` (score 0.606) |

**Por que `bge-m3`:**

- Multilíngue de verdade (inclui PT-BR no treinamento).
- Mix de treinamento incluiu código.
- 1024 dims, 567M params — CPU dá conta tranquilo.
- Suporta dense + sparse no mesmo modelo, o que abre caminho para *hybrid search* sem trocar de modelo depois.

### 3. Dimensão do modelo é um contrato com a coleção

Coleção `csharp_code` foi criada com 768 dims (nomic). Coleção `csharp_code_bgem3` foi criada com 1024 dims (bge-m3). **Não dá para misturar** — uma busca com vetor de 1024 numa coleção de 768 dá erro, e vice-versa.

Convenção que estamos seguindo:
- Nome da coleção carrega o modelo: `csharp_code_bgem3`.
- Mantemos a coleção antiga (`csharp_code` com nomic) viva para comparar lado a lado se quiser. Não custa nada — é só um pouco de disco.

Quando trocar de modelo no futuro, **crie nova coleção**, indexe nela, compare com a antiga, e só depois apague a antiga se a nova for melhor. Reindexar in-place é tentação ruim.

### 4. Diagnóstico de retrieval ruim — três queries, não uma

Quando o RAG está retornando lixo, rode estas três queries para isolar a causa antes de mudar qualquer coisa:

1. **Query natural no idioma do usuário** (PT-BR aqui).
2. **A mesma query traduzida** para o idioma do código (EN).
3. **Uma query lexical** com termos que aparecem literalmente no código (`Remove`, `SaveChangesAsync`).

Mapeamento:

| Falha em... | Causa provável | Solução |
|---|---|---|
| Só PT-BR | Modelo monolíngue | Trocar por modelo multilíngue |
| PT-BR e EN | Modelo não entende código | Trocar por modelo code-aware |
| As três | Embedding ruim **ou** chunking ruim | Inspecionar chunks antes do embedding |
| Só lexical | Embeddings dense puros bastam | Não precisa hybrid; ignore |
| Lexical mas não semântico | Embeddings ruins, hybrid não resolve | Trocar modelo de embedding |

### 5. O que **não** consertamos ainda (por ser desnecessário no momento)

- **Hybrid search (dense + BM25)**: bge-m3 puro já passa as três queries. Adicionar BM25 só compensa se medirmos casos onde dense falha — não é um "sempre faça".
- **Reranker**: idem. Modelos pequenos *gostariam* de reranker, mas para 157 chunks e queries específicas o ganho é marginal. Adicionar quando o LLM começar a alucinar por causa de chunks irrelevantes no contexto.
- **Validação automática modelo↔coleção**: hoje você precisa lembrar de passar `--model bge-m3`. Bug latente: passar modelo errado pode dar 404 (Ollama) ou erro de dimensão (Qdrant). Vai entrar como melhoria pequena depois.

---

# Etapa 3 — LLM no loop

## Pré-requisitos

- Etapas 1 e 2 concluídas.
- Um LLM para código no Ollama, por exemplo:

  ```bash
  ollama pull qwen2.5-coder:7b
  ```

  Substitua por `gemma2:9b`, `codellama:7b` ou outro que você tenha. Passe via `--llm`.

## Perguntar

```bash
python ask.py "como deletar um TodoItem do banco?"
```

A resposta é **streamada** no terminal token a token. O modelo recebe os 5 chunks mais relevantes e tem instrução estrita de:
- citar `arquivo.cs:linha`
- dizer "Não encontrado nos trechos fornecidos." se a resposta não estiver nos chunks
- responder em PT-BR

Para inspecionar o que foi recuperado antes da resposta:

```bash
python ask.py "como deletar um TodoItem do banco?" --show-context
```

Trocar o LLM:

```bash
python ask.py "..." --llm gemma2:9b
```

Mais chunks no contexto (use com cuidado, modelos 7B perdem foco):

```bash
python ask.py "explique o pipeline de validação" --k 8
```

## Ajuste fino do prompt

O `SYSTEM_PROMPT` em [ask.py](ask.py) é a "cara" do assistente. Se o modelo:

- **Inventa nomes de classes** → reforce "não invente, use exatamente os nomes dos trechos".
- **Esquece de citar arquivo:linha** → adicione um exemplo de citação correta no system prompt.
- **Responde em inglês** → adicione "Responda SEMPRE em português" no início.
- **Divaga** → reduza `temperature` em [ask.py](ask.py) de `0.2` para `0.0`.

## O que ainda NÃO temos (melhorias futuras mensuráveis)

- **Hybrid search** (dense + sparse BM25): melhora retrieval quando a query usa termos exatos do código (nomes de método, flags). Qdrant suporta nativo.
- **Reranker** (Qwen3-Reranker / bge-reranker-v2-m3): pega top-30 do retrieval e corta para top-5 — eleva precisão para LLMs pequenos.
- **Indexação NL**: gerar uma descrição em PT-BR de cada chunk via LLM e embedar à parte; resolve queries muito abstratas.
- **Expansão de contexto**: ao retornar um chunk, puxar automaticamente vizinhos do mesmo arquivo (método anterior + próximo) para dar mais contexto ao LLM.
- **Re-chunk para métodos gigantes**: se algum método ultrapassar ~800 tokens, dividir antes de embedar.
- **Validar dim do modelo vs coleção**: o `search.py`/`ask.py` deveriam ler o modelo gravado no Qdrant em vez de exigir `--model` na linha de comando.

Implemente cada uma só **depois** de medir que é necessária — em projetos pequenos como esse, o pipeline atual já costuma resolver.
