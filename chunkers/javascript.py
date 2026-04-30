"""
JavaScript/JSX semantic chunker using Tree-sitter.

Parses .js and .jsx files and returns one Chunk per function, class,
class method, or exported arrow function.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tree_sitter import Language, Node, Parser

from chunkers.base import (
    SKIP_DIR_PARTS,
    SMALL_CLASS_LINE_LIMIT,
    Chunk,
    _extract_name,
    _extract_signature,
    _find_child,
    _make_chunk,
    _text,
)

try:
    import tree_sitter_javascript as tsjs
    try:
        _JS_LANGUAGE = Language(tsjs.language())
    except AttributeError:
        _JS_LANGUAGE = Language(tsjs.JAVASCRIPT)
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "tree-sitter-javascript not found. Run: pip install tree-sitter-javascript"
    ) from exc

_parser = Parser(_JS_LANGUAGE)

SKIP_DIR_PARTS_EXTRA = {"dist", "build", ".next", "coverage", ".turbo", "out"}
SKIP_FILE_SUFFIXES = (".min.js",)
SKIP_FILE_NAMES: set[str] = set()

_CLASS_NODES = {"class_declaration"}
_MEMBER_NODES = {"method_definition", "field_definition"}


def _is_arrow_export(node: Node, src: bytes) -> tuple[bool, str]:
    if node.type not in ("lexical_declaration", "variable_declaration"):
        return False, ""
    for child in node.children:
        if child.type == "variable_declarator":
            val = child.child_by_field_name("value")
            if val and val.type == "arrow_function":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return True, _text(name_node, src)
    return False, ""


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    src = path.read_bytes()
    tree = _parser.parse(src)
    root = tree.root_node

    imports = [
        _text(n, src).strip()
        for n in root.children
        if n.type == "import_statement"
    ]

    chunks: list[Chunk] = []

    def walk(node: Node, namespace: str, class_name: str) -> None:
        # Unwrap export statements
        if node.type == "export_statement":
            for child in node.children:
                if child.type not in ("export", "default", ";", "comment"):
                    walk(child, namespace, class_name)
            return

        # Classes
        if node.type in _CLASS_NODES:
            name = _extract_name(node, src)
            full_class = f"{class_name}.{name}" if class_name else name
            n_lines = node.end_point[0] - node.start_point[0] + 1

            if n_lines <= SMALL_CLASS_LINE_LIMIT:
                chunks.append(
                    _make_chunk(
                        node, namespace, full_class, full_class, "class",
                        _extract_signature(node, src), imports, src, path, repo_root,
                        language="javascript",
                    )
                )
                return

            body = node.child_by_field_name("body") or _find_child(node, "class_body")
            if body:
                for child in body.children:
                    walk(child, namespace, full_class)
            return

        # Functions
        if node.type in ("function_declaration", "generator_function_declaration"):
            name = _extract_name(node, src)
            symbol = f"{class_name}.{name}" if class_name else name
            chunks.append(
                _make_chunk(
                    node, namespace, class_name, symbol, "function",
                    _extract_signature(node, src), imports, src, path, repo_root,
                    language="javascript",
                )
            )
            return

        # Class methods and fields
        if node.type in _MEMBER_NODES:
            name = _extract_name(node, src)
            if not name or name == "<anon>":
                return
            kind = "constructor" if name == "constructor" else "method"
            symbol = f"{class_name}.{name}" if class_name else name
            chunks.append(
                _make_chunk(
                    node, namespace, class_name, symbol, kind,
                    _extract_signature(node, src), imports, src, path, repo_root,
                    language="javascript",
                )
            )
            return

        # Top-level exported arrow functions
        if not class_name:
            is_arrow, name = _is_arrow_export(node, src)
            if is_arrow:
                chunks.append(
                    _make_chunk(
                        node, namespace, class_name, name, "function",
                        _extract_signature(node, src), imports, src, path, repo_root,
                        language="javascript",
                    )
                )
                return

        for child in node.children:
            walk(child, namespace, class_name)

    walk(root, namespace="", class_name="")
    return chunks


def iter_js_files(root: Path):
    for pattern in ("*.js", "*.jsx"):
        for path in root.rglob(pattern):
            if any(part in SKIP_DIR_PARTS for part in path.parts):
                continue
            if any(part in SKIP_DIR_PARTS_EXTRA for part in path.parts):
                continue
            if path.name in SKIP_FILE_NAMES:
                continue
            if any(path.name.endswith(s) for s in SKIP_FILE_SUFFIXES):
                continue
            yield path


def chunk_repo(repo_path: Path) -> list[Chunk]:
    """Chunk all .js/.jsx files under repo_path."""
    repo_root = repo_path.resolve()
    chunks: list[Chunk] = []
    for js_file in iter_js_files(repo_root):
        try:
            chunks.extend(chunk_file(js_file, repo_root))
        except Exception as exc:
            print(f"warn: failed to parse {js_file}: {exc}", file=sys.stderr)
    return chunks
