"""
Stage 1 — Repository-level semantic summarization.

Walks one or more repositories, extracts high-level signals (README,
top-level structure, key manifests like package.json / pyproject.toml /
*.csproj / go.mod / Cargo.toml) and asks a local LLM to produce a structured
JSON summary per repository.

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
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

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
    "You are a senior software engineer cataloguing repositories. "
    "Given a repository's metadata (name, structure, README, manifests), "
    "produce ONE JSON object describing it.\n\n"
    "Reply with ONLY the JSON object — no preamble, no markdown fences, "
    "no commentary. Required keys:\n"
    "  - repository: string (use the provided repo name unless the README "
    "establishes a different canonical name)\n"
    "  - summary: 1-3 sentence high-level description of what the project does\n"
    "  - main_technologies: array of strings (languages, frameworks, key tooling)\n"
    "  - domain: short category (e.g. 'Web API', 'CLI tool', 'Data pipeline', "
    "'Infrastructure', 'Library')\n"
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


def _build_user_message(repo_root: Path) -> str:
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
    max_tokens: int = 768,
) -> dict:
    client = make_client(base_url)
    user_msg = _build_user_message(repo_root)

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
        content = (getattr(msg, "reasoning_content", None) or "").strip()

    parsed = _extract_json(content) or {}

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
        "--max-tokens", type=int, default=768, dest="max_tokens",
        help="Max tokens per summary response (default: 768)",
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
    for repo in tqdm(repos, desc="Summarizing", unit="repo"):
        try:
            summary = summarize_repository(repo, args.model, base_url, args.max_tokens)
        except Exception as exc:
            print(f"warn: failed to summarize {repo}: {exc}", file=sys.stderr)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
