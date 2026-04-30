"""
Diagnostic: prints the tree-sitter AST of a single .cs file with field names.

Usage:
    python inspect_ast.py <file.cs> [max_depth]
"""

import sys
from pathlib import Path

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

CSHARP = Language(tscsharp.language())
parser = Parser(CSHARP)


def dump(node, src: bytes, depth: int, max_depth: int, field: str | None = None) -> None:
    if depth > max_depth:
        return
    text = src[node.start_byte : node.end_byte].decode("utf-8", "replace")
    snippet = text.replace("\n", " ").replace("\r", "")[:60]
    field_str = f" ({field})" if field else ""
    print(f"{'  ' * depth}{node.type}{field_str}: {snippet!r}")

    cursor = node.walk()
    if cursor.goto_first_child():
        while True:
            dump(cursor.node, src, depth + 1, max_depth, cursor.field_name)
            if not cursor.goto_next_sibling():
                break


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    src = path.read_bytes()
    tree = parser.parse(src)
    dump(tree.root_node, src, 0, max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
