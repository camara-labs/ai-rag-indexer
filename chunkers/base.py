"""
Shared types and pure helper functions used by all language-specific chunkers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

SMALL_CLASS_LINE_LIMIT = 60
MAX_IMPORTS_IN_HEADER = 10
SKIP_DIR_PARTS = {"bin", "obj", ".git", "node_modules", "packages"}


@dataclass
class Chunk:
    code: str
    file_path: str
    namespace: str
    class_name: str
    symbol: str
    kind: str
    signature: str
    imports: list[str]
    start_line: int
    end_line: int
    content_hash: str
    language: str = ""


# ── AST helpers ────────────────────────────────────────────────────────────────

def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_child(node: Node, type_name: str) -> Node | None:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _extract_name(node: Node, src: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, src)
    ident = _find_child(node, "identifier")
    return _text(ident, src) if ident else "<anon>"


def _extract_signature(node: Node, src: bytes) -> str:
    body = (
        _find_child(node, "block")
        or _find_child(node, "statement_block")        # JS/TS function bodies
        or _find_child(node, "class_body")             # JS/TS class declarations
        or _find_child(node, "arrow_expression_clause")
        or _find_child(node, "declaration_list")
        or node.child_by_field_name("body")
    )
    end = body.start_byte if body else node.end_byte
    sig = src[node.start_byte : end].decode("utf-8", errors="replace").strip()
    if sig.endswith("{"):
        sig = sig[:-1].rstrip()
    if sig.endswith("=>"):
        sig = sig[:-2].rstrip()
    return sig


def _collect_leading_comments(node: Node) -> int:
    """Walks backwards from `node` to include adjacent comments / XML doc."""
    start = node.start_byte
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        start = prev.start_byte
        prev = prev.prev_sibling
    return start


def _make_chunk(
    node: Node,
    namespace: str,
    class_name: str,
    symbol: str,
    kind: str,
    signature: str,
    imports: list[str],
    src: bytes,
    path: Path,
    repo_root: Path,
    language: str = "",
) -> Chunk:
    code_start = _collect_leading_comments(node)
    raw_code = src[code_start : node.end_byte].decode("utf-8", errors="replace")

    rel_path = path.relative_to(repo_root).as_posix()

    header_lines = [f"// File: {rel_path}"]
    if namespace:
        header_lines.append(f"// Namespace: {namespace}")
    if class_name and kind != "class":
        header_lines.append(f"// Class: {class_name}")
    header_lines.extend(imports[:MAX_IMPORTS_IN_HEADER])
    header = "\n".join(header_lines) + "\n\n"

    full_code = header + raw_code
    content_hash = hashlib.sha1(full_code.encode("utf-8")).hexdigest()

    return Chunk(
        code=full_code,
        file_path=rel_path,
        namespace=namespace,
        class_name=class_name,
        symbol=symbol,
        kind=kind,
        signature=signature,
        imports=imports,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        content_hash=content_hash,
        language=language,
    )
