"""
Stage 1 — Repository-level semantic summarization.

Walks one or more repositories and, for each one, asks a local LLM to produce
a structured JSON summary. The prompt is built from:

  - README + top-level structure + key manifests
  - A code-structural digest derived from the existing AST chunkers in
    `chunkers/` (resources/modules/providers for Terraform; classes/
    interfaces/types/functions for TypeScript), aggregated into compact
    counts and name samples.

Only TypeScript and Terraform repositories are supported — these are the
fine-grained, single-purpose stacks where a per-repo summary actually adds
value. Repositories whose detected language has no chunker (e.g. C#, Python,
Java, Go) are reported as errors and skipped, and the script exits with a
non-zero status code if any repo failed.

Input forms (any combination):
- A single repository path
- A list of repository paths (positional args)
- A "container" folder — recursively walked until a directory containing
  `.git` is found; that directory is treated as a repository root and its
  subtree is not descended further.

Configuration (.env, padrão LLM_*):
    LLM_BASE_URL        — local LLM server (OpenAI-compatible)
    LLM_MODEL           — chat model id
    OPENAI_BASE_URL     — fallback for LLM_BASE_URL
    OPENAI_API_KEY      — any string for local servers

Output:
    --output FILE       — single JSON file with a list of summaries (default)
    --output-dir DIR    — one JSON file per repository

Usage:
    python repo_summarize.py /path/to/repo
    python repo_summarize.py /path/to/parent_folder
    python repo_summarize.py /a /b /c --output .repos/repos.json
    python repo_summarize.py /workspace --output-dir .repos/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from chunkers import chunk_repo, detect_languages
from llm_client import make_client

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

_README_NAMES = (
    "README.md", "README.MD", "Readme.md", "readme.md",
    "README", "README.txt", "README.rst",
)

_KEY_FILES = (
    "package.json", "tsconfig.json", "pyproject.toml", "requirements.txt",
    "setup.py", "setup.cfg", "Cargo.toml", "go.mod", "pom.xml",
    "build.gradle", "build.gradle.kts", "Gemfile", "composer.json",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "CMakeLists.txt",
)

_KEY_GLOBS = ("*.csproj", "*.sln", "*.fsproj", "*.vbproj", "*.tf")

_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "bin", "obj", ".next", "coverage", ".turbo", "out",
    ".terraform", ".idea", ".vscode", ".gradle", "target", "vendor",
    ".pytest_cache", ".mypy_cache",
}

_SYSTEM_PROMPT = (
    "You are a senior product engineer cataloguing repositories. "
    "Given a repository's metadata (name, structure, AST digest, manifests, "
    "README), produce ONE JSON object describing it.\n\n"
    "Reply with ONLY the JSON object — no preamble, no markdown fences, "
    "no commentary. Required keys:\n"
    "  - repository: string (use the provided repo name unless the README "
    "establishes a different canonical name)\n"
    "  - summary: 1-3 sentences describing the service's business behavior. "
    "Follow this structure: (1) what event or request triggers it and what "
    "input it carries; (2) what specific field or record it reads or mutates "
    "and under what condition; (3) the exact names of any events, responses, "
    "or side-effects it produces for each outcome. "
    "Use concrete names found in the source (event types, field names, "
    "status values) — do not replace them with generic terms like "
    "'subsequent actions' or 'relevant events'.\n"
    "  - main_technologies: array of strings (languages, frameworks, key tooling)\n"
    "  - domain: short category (e.g. 'Web API', 'CLI tool', 'Data pipeline', "
    "'Infrastructure', 'Library', 'Event-driven service')\n"
    "  - tags: array of short retrieval-oriented tags\n\n"
    "Be precise and grounded in the input. Do not invent technologies that "
    "are not visible in the manifests, structure, or README."
)
def _read_text(path: Path, max_chars: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _find_readme(repo_root: Path) -> tuple[str, str]:
    for name in _README_NAMES:
        cand = repo_root / name
        if cand.is_file():
            return cand.name, _read_text(cand)
    return "", ""


def _scan_structure(repo_root: Path, max_entries: int = 80) -> list[str]:
    """List top-level entries plus notable second-level entries."""
    top: list[str] = []
    second: list[str] = []
    try:
        children = sorted(repo_root.iterdir(), key=lambda p: p.name.lower())
    except Exception:
        return top

    for child in children:
        if child.is_dir() and child.name in _SKIP_DIRS:
            continue
        if child.name.startswith(".") and child.name not in {".github", ".gitlab"}:
            continue
        top.append(child.name + ("/" if child.is_dir() else ""))

        if child.is_dir() and child.name not in _SKIP_DIRS:
            try:
                for sub in sorted(child.iterdir(), key=lambda p: p.name.lower()):
                    if sub.name.startswith("."):
                        continue
                    if sub.is_dir() and sub.name in _SKIP_DIRS:
                        continue
                    second.append(f"{child.name}/{sub.name}" + ("/" if sub.is_dir() else ""))
            except Exception:
                pass

    return (top + second)[:max_entries]


def _key_file_signals(repo_root: Path) -> list[str]:
    found: list[str] = []
    for name in _KEY_FILES:
        if (repo_root / name).is_file():
            found.append(name)
    for pattern in _KEY_GLOBS:
        for p in repo_root.glob(pattern):
            if p.is_file():
                found.append(p.name)

    parts: list[str] = []
    parts.append("Key files: " + (", ".join(sorted(set(found))) if found else "(none detected)"))

    # Inline a few of the most informative manifests so the LLM can ground itself.
    for manifest, max_chars in (
        ("package.json", 2000),
        ("pyproject.toml", 2000),
        ("Cargo.toml", 1500),
        ("go.mod", 1000),
        ("requirements.txt", 1000),
    ):
        cand = repo_root / manifest
        if cand.is_file():
            parts.append(f"--- {manifest} ---\n{_read_text(cand, max_chars)}")

    # First .csproj is enough to identify the .NET stack.
    csproj = next(iter(repo_root.glob("*.csproj")), None)
    if csproj:
        parts.append(f"--- {csproj.name} ---\n{_read_text(csproj, 1500)}")

    return parts


_MAIN_BRANCHES = {"main", "master"}


def _git_info(repo_root: Path) -> tuple[str, str]:
    """Return (branch, short_commit) for the repo at repo_root.

    Returns ("", "") on any failure (no commits yet, not a git repo, etc.).
    """
    def _run(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "rev-parse", "HEAD"])
    return branch, commit


def _walk_for_repos(root: Path, out: list[Path]) -> None:
    """Recurse pruning at repo boundaries: stop descending into a found repo."""
    if (root / ".git").exists():
        out.append(root)
        return
    try:
        children = list(root.iterdir())
    except (PermissionError, OSError):
        return
    for child in children:
        if not child.is_dir():
            continue
        if child.name in _SKIP_DIRS or child.name.startswith("."):
            continue
        _walk_for_repos(child, out)


def discover_repositories(paths: list[Path]) -> list[Path]:
    """Resolve input paths into a deduplicated, sorted list of repository roots."""
    seen: set[Path] = set()
    roots: list[Path] = []

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            roots.append(rp)

    for raw in paths:
        p = raw.expanduser()
        try:
            p = p.resolve()
        except OSError:
            print(f"warn: cannot resolve {raw} — skipping", file=sys.stderr)
            continue
        if not p.exists() or not p.is_dir():
            print(f"warn: {p} is not a directory — skipping", file=sys.stderr)
            continue
        if (p / ".git").exists():
            _add(p)
            continue
        found: list[Path] = []
        _walk_for_repos(p, found)
        for r in found:
            _add(r)

    return sorted(roots)


# ── Chunk-based structural digest ────────────────────────────────────────────
# Only TypeScript and Terraform are supported for repository-level summarization
# — they have fine-grained semantic units that aggregate well, and their repos
# are typically small and focused. C# repos already carry full context in a
# single artifact, so they are intentionally excluded.

_DIGEST_LANGUAGES: tuple[str, ...] = ("typescript", "terraform")


class UnsupportedRepositoryError(ValueError):
    """Raised when a repository has no language supported for summarization."""


def _truncated_list(items: list[str], n: int = 15) -> str:
    head = items[:n]
    suffix = ", ..." if len(items) > n else ""
    return ", ".join(head) + suffix


def _digest_typescript(chunks, pkg_info: dict | None = None) -> str:
    if not chunks:
        return "TypeScript: (no chunks extracted)"

    files = sorted({ch.file_path for ch in chunks})
    by_dir: Counter = Counter()
    for f in files:
        parts = f.split("/")
        head = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        by_dir[head] += 1

    kinds = Counter(ch.kind for ch in chunks)
    classes = sorted({ch.symbol for ch in chunks if ch.kind == "class"})
    interfaces = sorted({ch.symbol for ch in chunks if ch.kind == "interface"})
    types = sorted({ch.symbol for ch in chunks if ch.kind == "type"})
    functions = sorted({ch.symbol for ch in chunks if ch.kind == "function"})

    lines = [f"TypeScript ({len(chunks)} chunks across {len(files)} files)"]
    if by_dir:
        lines.append("  Files by directory:")
        for d, n in by_dir.most_common(8):
            lines.append(f"    - {d}: {n}")
    if kinds:
        lines.append("  Kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    if classes:
        lines.append(f"  Classes: {_truncated_list(classes)}")
    if interfaces:
        lines.append(f"  Interfaces: {_truncated_list(interfaces)}")
    if types:
        lines.append(f"  Types: {_truncated_list(types)}")
    if functions:
        lines.append(f"  Functions: {_truncated_list(functions)}")

    if pkg_info:
        def _fmt_deps(deps: dict) -> str:
            return ", ".join(f"{k}@{v}" for k, v in deps.items())

        deps = pkg_info.get("dependencies") or {}
        dev = pkg_info.get("devDependencies") or {}
        peer = pkg_info.get("peerDependencies") or {}
        if deps:
            lines.append(f"  Dependencies ({len(deps)}): {_fmt_deps(deps)}")
        if dev:
            lines.append(f"  Dev dependencies ({len(dev)}): {_fmt_deps(dev)}")
        if peer:
            lines.append(f"  Peer dependencies ({len(peer)}): {_fmt_deps(peer)}")

    return "\n".join(lines)


def _digest_terraform(chunks) -> str:
    if not chunks:
        return "Terraform: (no chunks extracted)"

    files = sorted({ch.file_path for ch in chunks})
    resources = Counter(
        ch.class_name for ch in chunks if ch.kind == "resource" and ch.class_name
    )
    data_sources = Counter(
        ch.class_name for ch in chunks if ch.kind == "data" and ch.class_name
    )
    modules = sorted({ch.symbol for ch in chunks if ch.kind == "module"})
    providers = sorted({ch.symbol for ch in chunks if ch.kind == "provider"})
    variables = sorted({ch.symbol for ch in chunks if ch.kind == "variable"})
    outputs = sorted({ch.symbol for ch in chunks if ch.kind == "output"})

    lines = [f"Terraform ({len(chunks)} blocks across {len(files)} files)"]
    if resources:
        lines.append("  Resources: " + ", ".join(
            f"{rt}×{n}" for rt, n in resources.most_common()
        ))
    if data_sources:
        lines.append("  Data sources: " + ", ".join(
            f"{rt}×{n}" for rt, n in data_sources.most_common()
        ))
    if modules:
        lines.append(f"  Modules: {_truncated_list(modules)}")
    if providers:
        lines.append(f"  Providers: {', '.join(providers)}")
    if variables:
        lines.append(f"  Variables: {_truncated_list(variables)}")
    if outputs:
        lines.append(f"  Outputs: {_truncated_list(outputs)}")
    return "\n".join(lines)


_DIGEST_BUILDERS = {
    "typescript": _digest_typescript,
    "terraform":  _digest_terraform,
}


def build_chunk_digest(repo_root: Path) -> tuple[str, list[str]]:
    """Run the existing chunkers and aggregate their output into a compact digest.

    Returns
    -------
    (digest_text, languages)
        ``digest_text`` is the multi-section string ready to feed the LLM,
        ``languages`` is the list of supported languages actually present.

    Raises
    ------
    UnsupportedRepositoryError
        If the repository has no language supported for summarization
        (only TypeScript and Terraform are supported).
    """
    detected = detect_languages(repo_root)
    if not detected:
        raise UnsupportedRepositoryError(
            "no supported source files were detected in the repository"
        )

    supported = [lang for lang in detected if lang in _DIGEST_LANGUAGES]
    if not supported:
        raise UnsupportedRepositoryError(
            f"no chunker is available for the languages detected "
            f"({', '.join(detected)}); supported: {', '.join(_DIGEST_LANGUAGES)}"
        )

    sections: list[str] = []
    for lang in supported:
        chunks = chunk_repo(language=lang, repo_path=repo_root)
        if lang == "typescript":
            from chunkers.typescript import extract_package_info
            sections.append(_digest_typescript(chunks, extract_package_info(repo_root)))
        else:
            sections.append(_DIGEST_BUILDERS[lang](chunks))

    return "\n\n".join(sections), supported


def _build_user_message(repo_root: Path, digest: str) -> str:
    readme_name, readme = _find_readme(repo_root)
    structure = _scan_structure(repo_root)
    key_signals = _key_file_signals(repo_root)

    parts: list[str] = []
    parts.append(f"Repository name: {repo_root.name}")
    parts.append(f"Repository path: {repo_root}")
    parts.append("")
    parts.append("Top-level structure:")
    parts.append("\n".join(f"- {e}" for e in structure) if structure else "- (empty)")
    parts.append("")
    parts.append("Code-structural digest (from AST):")
    parts.append(digest)
    parts.append("")
    parts.extend(key_signals)
    parts.append("")
    if readme:
        parts.append(f"--- {readme_name} ---")
        parts.append(readme)
    else:
        parts.append("(no README detected)")
    return "\n".join(parts)


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = _THINK_RE.sub("", text or "").strip()
    # Handle truncated <think> block (model ran out of tokens mid-reasoning).
    if text.startswith("<think>"):
        brace = text.find("{")
        text = text[brace:] if brace != -1 else ""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Scan for the first balanced top-level object.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def summarize_repository(
    repo_root: Path,
    model: str,
    base_url: str | None = None,
    max_tokens: int = 4096,
    verbose: bool = False,
) -> dict:
    digest, languages = build_chunk_digest(repo_root)

    client = make_client(base_url)
    user_msg = _build_user_message(repo_root, digest)

    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        extra_body={"thinking_budget": 0},
    )
    duration_s = round(time.monotonic() - t0, 2)

    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = (getattr(msg, "reasoning_content", None) or "").strip()

    if verbose:
        tqdm.write(f"\n--- raw LLM response ({repo_root.name}) ---\n{content}\n---")

    parsed = _extract_json(content)
    if parsed is None:
        tqdm.write(
            f"warn: could not extract JSON from LLM response for {repo_root.name}. "
            f"Raw response ({len(content)} chars):\n{content[:500]}"
            + ("..." if len(content) > 500 else ""),
            file=sys.stderr,
        )
        parsed = {}

    summary = {
        "repository": (parsed.get("repository") or repo_root.name).strip(),
        "path": str(repo_root),
        "summary": (parsed.get("summary") or "").strip(),
        "main_technologies": list(parsed.get("main_technologies") or []),
        "domain": (parsed.get("domain") or "").strip(),
        "tags": list(parsed.get("tags") or []),
    }
    blob = json.dumps(summary, sort_keys=True, ensure_ascii=False)
    summary["content_hash"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    branch, commit = _git_info(repo_root)
    summary["git_branch"] = branch
    summary["git_commit"] = commit
    summary["summarized_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    usage = resp.usage
    summary["llm_stats"] = {
        "model":             resp.model or model,
        "prompt_tokens":     usage.prompt_tokens     if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens":      usage.total_tokens      if usage else None,
        "duration_s":        duration_s,
    }

    if branch and branch not in _MAIN_BRANCHES:
        tqdm.write(
            f"warn: {repo_root.name} is on branch '{branch}', not main/master — "
            "summary may not reflect production code",
            file=sys.stderr,
        )

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a JSON summary per repository using a local LLM."
    )
    ap.add_argument(
        "paths", nargs="+", type=Path,
        help="One or more paths. Each may be a repo root, or a parent folder "
             "that contains repositories (recursively discovered via .git).",
    )
    ap.add_argument(
        "--output", type=Path, default=Path(".repos/repos.json"),
        help="Single JSON file with all summaries (default: .repos/repos.json)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None, dest="output_dir",
        help="If set, write one JSON file per repository in this directory.",
    )
    ap.add_argument(
        "--model", default=DEFAULT_LLM_MODEL or None,
        help="Chat model id (default: LLM_MODEL from .env)",
    )
    ap.add_argument(
        "--base-url", default=None, dest="base_url",
        help="LLM server base URL (default: LLM_BASE_URL from .env)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=4096, dest="max_tokens",
        help="Max tokens per summary response (default: 4096; "
             "thinking/reasoning models need ≥2048 for reasoning + JSON output)",
    )
    ap.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print the raw LLM response for each repository (useful for debugging "
             "JSON extraction failures)",
    )
    args = ap.parse_args()

    if not args.model:
        print(
            "error: no LLM model configured. "
            "Set LLM_MODEL in .env or pass --model.",
            file=sys.stderr,
        )
        return 1

    base_url = args.base_url or DEFAULT_LLM_BASE_URL
    repos = discover_repositories(args.paths)
    if not repos:
        print("No repositories found.", file=sys.stderr)
        return 1

    print(f"Discovered {len(repos)} repositor{'y' if len(repos) == 1 else 'ies'}:")
    for r in repos:
        print(f"  - {r}")
    print(f"Model  : {args.model}")
    print(f"Server : {base_url}")
    print()

    summaries: list[dict] = []
    errors = 0
    for repo in tqdm(repos, desc="Summarizing", unit="repo"):
        try:
            summary = summarize_repository(
                repo, args.model, base_url, args.max_tokens, verbose=args.verbose
            )
        except UnsupportedRepositoryError as exc:
            tqdm.write(f"error: {repo}: {exc}")
            errors += 1
            continue
        except Exception as exc:
            tqdm.write(f"warn: failed to summarize {repo}: {exc}")
            errors += 1
            continue
        summaries.append(summary)

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", summary["repository"])
            (args.output_dir / f"{safe}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if not args.output_dir:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {len(summaries)} summaries -> {args.output}")
    else:
        print(f"\nWrote {len(summaries)} summary files -> {args.output_dir}")

    if errors:
        print(
            f"{errors} repositor{'y' if errors == 1 else 'ies'} skipped "
            f"(unsupported language or processing error).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
