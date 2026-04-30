"""
Terraform (HCL) semantic chunker using Tree-sitter.

Parses .tf files and returns one Chunk per top-level HCL block
(resource, data, module, variable, output, locals, provider, terraform).
"""

from __future__ import annotations

import sys
from pathlib import Path

from tree_sitter import Language, Node, Parser

from chunkers.base import (
    SKIP_DIR_PARTS,
    Chunk,
    _find_child,
    _make_chunk,
    _text,
)

try:
    import tree_sitter_hcl as tshcl
    try:
        _HCL_LANGUAGE = Language(tshcl.language())
    except AttributeError:
        _HCL_LANGUAGE = Language(tshcl.HCL)
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "tree-sitter-hcl not found. Run: pip install tree-sitter-hcl"
    ) from exc

_parser = Parser(_HCL_LANGUAGE)

SKIP_DIR_PARTS_EXTRA = {".terraform"}
SKIP_FILE_SUFFIXES = (".tfstate", ".tfstate.backup")
SKIP_FILE_NAMES: set[str] = set()

_BLOCK_TYPES = {
    "resource", "data", "module", "variable", "output",
    "locals", "provider", "terraform",
}


def _string_label(node: Node, src: bytes) -> str:
    """Extract the string value of a quoted label node."""
    raw = _text(node, src).strip()
    # Try inner template_literal / quoted_template first
    for child in node.children:
        if child.type in ("template_literal", "quoted_template_start"):
            continue
        inner = _text(child, src).strip()
        if inner and inner not in ('"', "'"):
            return inner
    # Fall back: strip surrounding quotes from raw text
    return raw.strip('"\'')


def _collect_block_labels(node: Node, src: bytes) -> list[str]:
    """Collect all string label children of a block node."""
    labels: list[str] = []
    for child in node.children:
        if child.type in ("string_lit", "string_literal", "quoted_template"):
            labels.append(_string_label(child, src))
        elif child.type == "heredoc_template":
            labels.append(_text(child, src).strip())
    return labels


def _build_symbol(block_type: str, labels: list[str]) -> tuple[str, str]:
    """Return (symbol, class_name) for a block."""
    if block_type in ("resource", "data") and len(labels) >= 2:
        return f"{labels[0]}.{labels[1]}", labels[0]
    if labels:
        return labels[0], ""
    return block_type, ""


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    src = path.read_bytes()
    tree = _parser.parse(src)
    root = tree.root_node

    # namespace = relative directory (Terraform modules are directory-scoped)
    rel_dir = path.parent.relative_to(repo_root).as_posix()
    namespace = "" if rel_dir == "." else rel_dir

    chunks: list[Chunk] = []

    # HCL body node may be a direct child of root or root itself
    body = _find_child(root, "body") or root

    for node in body.children:
        node_type = node.type
        # Handle both "block" and direct block-type identifiers
        if node_type != "block":
            continue

        # First identifier child = block type keyword
        block_type_node = None
        for child in node.children:
            if child.type in ("identifier", "block_identifier"):
                block_type_node = child
                break

        if block_type_node is None:
            continue

        block_type = _text(block_type_node, src).strip()
        if block_type not in _BLOCK_TYPES:
            continue

        labels = _collect_block_labels(node, src)
        symbol, class_name = _build_symbol(block_type, labels)

        label_str = " ".join(f'"{l}"' for l in labels)
        signature = f'{block_type} {label_str}'.strip()

        chunks.append(
            _make_chunk(
                node, namespace, class_name, symbol, block_type,
                signature, [], src, path, repo_root,
                language="terraform",
            )
        )

    return chunks


def iter_tf_files(root: Path):
    for path in root.rglob("*.tf"):
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            continue
        if any(part in SKIP_DIR_PARTS_EXTRA for part in path.parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        yield path


def chunk_repo(repo_path: Path) -> list[Chunk]:
    """Chunk all .tf files under repo_path."""
    repo_root = repo_path.resolve()
    chunks: list[Chunk] = []
    for tf_file in iter_tf_files(repo_root):
        try:
            chunks.extend(chunk_file(tf_file, repo_root))
        except Exception as exc:
            print(f"warn: failed to parse {tf_file}: {exc}", file=sys.stderr)
    return chunks
