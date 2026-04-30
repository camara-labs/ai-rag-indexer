"""
Language dispatch for the chunking step.

Usage:
    from chunkers import chunk_repo
    chunks = chunk_repo(language="csharp", repo_path=Path("/path/to/repo"))

Supported languages:
    - "csharp"

Future languages (not yet implemented):
    - "nodejs" / "typescript"
"""

from __future__ import annotations

from pathlib import Path

from chunkers.base import Chunk

_SUPPORTED = ("csharp",)


def chunk_repo(language: str, repo_path: Path) -> list[Chunk]:
    """
    Chunk all source files of the given language under repo_path.

    Parameters
    ----------
    language:
        Source language to parse. Currently only ``"csharp"`` is supported.
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
        from chunkers.csharp import chunk_repo as _csharp_chunk_repo
        return _csharp_chunk_repo(repo_path)

    raise ValueError(
        f"Unsupported language: {language!r}. "
        f"Supported: {', '.join(_SUPPORTED)}"
    )


__all__ = ["chunk_repo", "Chunk"]
