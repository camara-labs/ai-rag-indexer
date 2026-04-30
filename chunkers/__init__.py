"""
Language dispatch for the chunking step.

Usage:
    from chunkers import chunk_repo, detect_languages
    languages = detect_languages(Path("/path/to/repo"))
    chunks = chunk_repo(language="typescript", repo_path=Path("/path/to/repo"))

Supported languages:
    - "csharp"
    - "typescript"
    - "javascript"
    - "terraform"
"""

from __future__ import annotations

from pathlib import Path

from chunkers.base import Chunk

_SUPPORTED = ("csharp", "javascript", "terraform", "typescript")

# Extensions per language used for auto-detection
_LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "csharp":     [".cs"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "terraform":  [".tf"],
}

# Directories to skip during language detection (union of all chunker skip rules)
_DETECT_SKIP_DIRS = {
    "bin", "obj", ".git", "node_modules", "packages",
    "dist", "build", ".next", "coverage", ".turbo", "out",
    ".terraform", "Migrations",
}

# File suffixes that don't count as real source files
_DETECT_SKIP_SUFFIXES: dict[str, tuple[str, ...]] = {
    "csharp":     (".Designer.cs", ".g.cs", ".g.i.cs"),
    "typescript": (".d.ts",),
    "javascript": (".min.js",),
    "terraform":  (),
}


def _has_source_files(root: Path, exts: list[str], skip_suffixes: tuple[str, ...]) -> bool:
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            if any(part in _DETECT_SKIP_DIRS for part in p.parts):
                continue
            if skip_suffixes and any(p.name.endswith(s) for s in skip_suffixes):
                continue
            return True
    return False


def detect_languages(repo_path: Path) -> list[str]:
    """Scan repo_path and return the list of languages that have source files.

    Languages are returned in a stable order matching ``_SUPPORTED``.
    Returns an empty list if no known source files are found.
    """
    root = Path(repo_path).resolve()
    detected = []
    for lang in _SUPPORTED:
        exts = _LANGUAGE_EXTENSIONS[lang]
        skip_suffixes = _DETECT_SKIP_SUFFIXES[lang]
        if _has_source_files(root, exts, skip_suffixes):
            detected.append(lang)
    return detected


def chunk_repo(language: str, repo_path: Path) -> list[Chunk]:
    """
    Chunk all source files of the given language under repo_path.

    Parameters
    ----------
    language:
        Source language to parse. Supported: ``"csharp"``, ``"typescript"``,
        ``"javascript"``, ``"terraform"``.
    repo_path:
        Absolute path to the root of the repository to scan.

    Returns
    -------
    list[Chunk]
        Flat list of Chunk objects with code + structural metadata.

    Raises
    ------
    ValueError
        If ``language`` is not supported.
    """
    lang = language.lower().strip()

    if lang == "csharp":
        from chunkers.csharp import chunk_repo as _impl
        return _impl(repo_path)

    if lang == "typescript":
        from chunkers.typescript import chunk_repo as _impl
        return _impl(repo_path)

    if lang == "javascript":
        from chunkers.javascript import chunk_repo as _impl
        return _impl(repo_path)

    if lang == "terraform":
        from chunkers.terraform import chunk_repo as _impl
        return _impl(repo_path)

    raise ValueError(
        f"Unsupported language: {language!r}. "
        f"Supported: {', '.join(_SUPPORTED)}"
    )


__all__ = ["chunk_repo", "detect_languages", "Chunk"]
