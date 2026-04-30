"""
C# semantic chunker using Tree-sitter.

Parses .cs files from a repository directory and returns a list of Chunk
objects — one per method, constructor, property, or small class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Node, Parser

from chunkers.base import (
    Chunk,
    SKIP_DIR_PARTS,
    _extract_name,
    _extract_signature,
    _find_child,
    _make_chunk,
    _text,
)

CSHARP = Language(tscsharp.language())
_parser = Parser(CSHARP)

SMALL_CLASS_LINE_LIMIT = 60

# Directories that only contain generated / build artefacts
SKIP_DIR_PARTS_EXTRA = {"Migrations"}

# File name suffixes produced by code generators (checked with str.endswith)
SKIP_FILE_SUFFIXES = (".Designer.cs", ".g.cs", ".g.i.cs")

# Exact file names that carry no semantic content
SKIP_FILE_NAMES = {"AssemblyInfo.cs", "GlobalUsings.g.cs"}


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    src = path.read_bytes()
    tree = _parser.parse(src)
    root = tree.root_node

    imports = [_text(n, src).strip() for n in root.children if n.type == "using_directive"]

    chunks: list[Chunk] = []

    def walk(node: Node, namespace: str, class_name: str) -> None:
        if node.type == "compilation_unit":
            current_ns = namespace
            for child in node.children:
                if child.type == "file_scoped_namespace_declaration":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        current_ns = _text(name_node, src)
                    continue
                walk(child, current_ns, class_name)
            return

        if node.type == "namespace_declaration":
            name_node = node.child_by_field_name("name")
            new_ns = _text(name_node, src) if name_node else namespace
            body = node.child_by_field_name("body") or _find_child(node, "declaration_list")
            if body is not None:
                for child in body.children:
                    walk(child, new_ns, class_name)
            return

        if node.type in (
            "class_declaration",
            "record_declaration",
            "struct_declaration",
            "interface_declaration",
        ):
            name = _extract_name(node, src)
            full_class = f"{class_name}.{name}" if class_name else name
            n_lines = node.end_point[0] - node.start_point[0] + 1

            if n_lines <= SMALL_CLASS_LINE_LIMIT:
                chunks.append(
                    _make_chunk(
                        node,
                        namespace,
                        full_class,
                        full_class,
                        "class",
                        _extract_signature(node, src),
                        imports,
                        src,
                        path,
                        repo_root,
                        language="csharp",
                    )
                )
                return

            body = node.child_by_field_name("body") or _find_child(node, "declaration_list")
            if body is not None:
                for child in body.children:
                    walk(child, namespace, full_class)
            return

        if node.type in (
            "method_declaration",
            "constructor_declaration",
            "property_declaration",
        ):
            name = _extract_name(node, src)
            kind = node.type.replace("_declaration", "")
            symbol = f"{class_name}.{name}" if class_name else name
            chunks.append(
                _make_chunk(
                    node,
                    namespace,
                    class_name,
                    symbol,
                    kind,
                    _extract_signature(node, src),
                    imports,
                    src,
                    path,
                    repo_root,
                    language="csharp",
                )
            )
            return

        for child in node.children:
            walk(child, namespace, class_name)

    walk(root, namespace="", class_name="")
    return chunks


def iter_cs_files(root: Path):
    for path in root.rglob("*.cs"):
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            continue
        if any(part in SKIP_DIR_PARTS_EXTRA for part in path.parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        if any(path.name.endswith(suffix) for suffix in SKIP_FILE_SUFFIXES):
            continue
        yield path


def chunk_repo(repo_path: Path) -> list[Chunk]:
    """Chunk all .cs files under repo_path. Returns a flat list of Chunk objects."""
    repo_root = repo_path.resolve()
    chunks: list[Chunk] = []
    for cs_file in iter_cs_files(repo_root):
        try:
            chunks.extend(chunk_file(cs_file, repo_root))
        except Exception as exc:
            print(f"warn: failed to parse {cs_file}: {exc}", file=sys.stderr)
    return chunks
