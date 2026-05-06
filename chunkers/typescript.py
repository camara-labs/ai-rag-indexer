"""
TypeScript/TSX semantic chunker using Tree-sitter.

Parses .ts and .tsx files and returns one Chunk per function, class,
class method, interface, type alias, or exported arrow function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tree_sitter import Language, Node, Parser

from chunkers.base import (
    SKIP_DIR_PARTS,
    SMALL_CLASS_LINE_LIMIT,
    Chunk,
    _collect_leading_comments,
    _extract_name,
    _extract_signature,
    _find_child,
    _make_chunk,
    _text,
)

try:
    import tree_sitter_typescript as tsts
    try:
        _TS_LANGUAGE = Language(tsts.language_typescript())
    except AttributeError:
        _TS_LANGUAGE = Language(tsts.language())
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "tree-sitter-typescript not found. Run: pip install tree-sitter-typescript"
    ) from exc

_parser = Parser(_TS_LANGUAGE)

SKIP_DIR_PARTS_EXTRA = {"dist", "build", ".next", "coverage", ".turbo", "out"}
SKIP_FILE_SUFFIXES = (".d.ts",)
SKIP_FILE_NAMES: set[str] = set()

_CLASS_NODES = {"class_declaration", "abstract_class_declaration"}
_MEMBER_NODES = {"method_definition", "public_field_definition", "property_definition"}


def _is_arrow_export(node: Node, src: bytes) -> tuple[bool, str]:
    """Return (True, name) if node is a top-level `const name = () => ...`."""
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
        if n.type in ("import_statement", "import_declaration")
    ]

    chunks: list[Chunk] = []

    def walk(node: Node, namespace: str, class_name: str) -> None:
        # Unwrap export statements to get the inner declaration
        if node.type == "export_statement":
            for child in node.children:
                if child.type not in ("export", "default", ";", "comment", "type", "declare"):
                    walk(child, namespace, class_name)
            return

        # TypeScript namespace / internal module
        if node.type in ("module", "internal_module", "namespace_declaration"):
            name_node = node.child_by_field_name("name")
            new_ns = _text(name_node, src) if name_node else namespace
            body = node.child_by_field_name("body") or _find_child(node, "statement_block")
            if body:
                for child in body.children:
                    walk(child, new_ns, class_name)
            return

        # Classes (including abstract)
        if node.type in _CLASS_NODES:
            name = _extract_name(node, src)
            full_class = f"{class_name}.{name}" if class_name else name
            n_lines = node.end_point[0] - node.start_point[0] + 1

            if n_lines <= SMALL_CLASS_LINE_LIMIT:
                chunks.append(
                    _make_chunk(
                        node, namespace, full_class, full_class, "class",
                        _extract_signature(node, src), imports, src, path, repo_root,
                        language="typescript",
                    )
                )
                return

            body = node.child_by_field_name("body") or _find_child(node, "class_body")
            if body:
                for child in body.children:
                    walk(child, namespace, full_class)
            return

        # Interfaces
        if node.type == "interface_declaration":
            name = _extract_name(node, src)
            symbol = f"{class_name}.{name}" if class_name else name
            chunks.append(
                _make_chunk(
                    node, namespace, class_name, symbol, "interface",
                    _extract_signature(node, src), imports, src, path, repo_root,
                    language="typescript",
                )
            )
            return

        # Type aliases
        if node.type == "type_alias_declaration":
            name = _extract_name(node, src)
            symbol = f"{class_name}.{name}" if class_name else name
            sig = _text(node, src).split("=")[0].strip()
            chunks.append(
                _make_chunk(
                    node, namespace, class_name, symbol, "type",
                    sig, imports, src, path, repo_root,
                    language="typescript",
                )
            )
            return

        # Functions
        if node.type in ("function_declaration", "generator_function_declaration"):
            name = _extract_name(node, src)
            symbol = f"{class_name}.{name}" if class_name else name
            chunks.append(
                _make_chunk(
                    node, namespace, class_name, symbol, "function",
                    _extract_signature(node, src), imports, src, path, repo_root,
                    language="typescript",
                )
            )
            return

        # Class methods and fields (inside large classes)
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
                    language="typescript",
                )
            )
            return

        # Top-level exported arrow functions: const foo = () => {}
        if not class_name:
            is_arrow, name = _is_arrow_export(node, src)
            if is_arrow:
                chunks.append(
                    _make_chunk(
                        node, namespace, class_name, name, "function",
                        _extract_signature(node, src), imports, src, path, repo_root,
                        language="typescript",
                    )
                )
                return

        for child in node.children:
            walk(child, namespace, class_name)

    walk(root, namespace="", class_name="")
    return chunks


def iter_ts_files(root: Path):
    for pattern in ("*.ts", "*.tsx"):
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
    """Chunk all .ts/.tsx files under repo_path."""
    repo_root = repo_path.resolve()
    chunks: list[Chunk] = []
    for ts_file in iter_ts_files(repo_root):
        try:
            chunks.extend(chunk_file(ts_file, repo_root))
        except Exception as exc:
            print(f"warn: failed to parse {ts_file}: {exc}", file=sys.stderr)
    return chunks


def extract_package_info(repo_path: Path) -> dict | None:
    """Parse the root package.json and return structured dependency info.

    Returns a dict with keys ``name``, ``version``, ``dependencies``,
    ``devDependencies``, and ``peerDependencies`` (each a ``{pkg: version}``
    mapping), or ``None`` if package.json is absent or unparseable.
    """
    pkg_file = Path(repo_path).resolve() / "package.json"
    if not pkg_file.is_file():
        return None
    try:
        data = json.loads(pkg_file.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {
        "name":             data.get("name", ""),
        "version":          data.get("version", ""),
        "dependencies":     data.get("dependencies") or {},
        "devDependencies":  data.get("devDependencies") or {},
        "peerDependencies": data.get("peerDependencies") or {},
    }
