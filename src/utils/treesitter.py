"""Tree-sitter parsing helpers for the dataset pipeline.

Wraps the tree_sitter_language_pack binding behind one class:

  * binding tolerance — the packaged binding exposes node fields either as
    methods (node.kind(), node.start_byte(), ...) or as attributes, and
    parses either str or bytes; every accessor here handles both styles;
  * function-definition extraction — names and bodies of every function /
    method / constructor definition in a source blob;
  * repository definition index — name -> [{file, function}] over a set of
    checked-out source files, used to resolve call-graph neighbor bodies.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tree_sitter_language_pack import get_parser

DEF_NODE_TYPES = {"function_definition", "method_declaration",
                  "constructor_declaration"}


class TreeSitter:
    """Language-aware source parsing (definitions, names, bodies)."""

    # ---- binding-tolerant node accessors ------------------------------ #
    @staticmethod
    def _call(v):
        return v() if callable(v) else v

    @classmethod
    def kind(cls, n) -> str:
        k = getattr(n, "kind", None)
        if k is None:
            k = getattr(n, "type", None)
        return cls._call(k)

    @classmethod
    def start_byte(cls, n) -> int:
        return cls._call(getattr(n, "start_byte"))

    @classmethod
    def end_byte(cls, n) -> int:
        return cls._call(getattr(n, "end_byte"))

    @staticmethod
    def field(n, name):
        return n.child_by_field_name(name)

    @classmethod
    def named_children(cls, n) -> List[Any]:
        cnt = cls._call(getattr(n, "named_child_count"))
        return [n.named_child(i) for i in range(cnt)]

    @classmethod
    def root(cls, tree):
        return cls._call(getattr(tree, "root_node"))

    @staticmethod
    def parser(lang_key: str):
        return get_parser(lang_key)

    @classmethod
    def parse(cls, parser, data: bytes):
        text = data.decode("utf-8", "ignore")
        try:
            return parser.parse(text), text.encode("utf-8")
        except TypeError:
            enc = text.encode("utf-8")
            return parser.parse(enc), enc

    @classmethod
    def text(cls, node, data: bytes) -> str:
        return data[cls.start_byte(node):cls.end_byte(node)].decode("utf-8", "ignore")

    @classmethod
    def walk(cls, node):
        stack = [node]
        while stack:
            cur = stack.pop()
            yield cur
            stack.extend(cls.named_children(cur))

    # ---- function-definition extraction ------------------------------- #
    @classmethod
    def def_name(cls, node, data: bytes, lang_key: str) -> str | None:
        k = cls.kind(node)
        if k in ("method_declaration", "constructor_declaration"):
            nm = cls.field(node, "name")
            return cls.text(nm, data) if nm is not None else None
        if k == "function_definition":
            if lang_key == "python":
                nm = cls.field(node, "name")
                return cls.text(nm, data) if nm is not None else None
            # C / C++: unwrap declarator chain to the function_declarator's name
            d = cls.field(node, "declarator")
            guard = 0
            while d is not None and cls.kind(d) in (
                    "pointer_declarator", "parenthesized_declarator",
                    "reference_declarator") and guard < 8:
                inner = cls.field(d, "declarator")
                d = inner if inner is not None else (
                    cls.named_children(d)[0] if cls.named_children(d) else None)
                guard += 1
            if d is not None and cls.kind(d) == "function_declarator":
                inner = cls.field(d, "declarator")
                if inner is not None:
                    ik = cls.kind(inner)
                    if ik in ("qualified_identifier", "field_identifier",
                              "identifier", "destructor_name", "operator_name"):
                        ids = [c for c in cls.walk(inner)
                               if cls.kind(c) == "identifier"]
                        return cls.text(ids[-1], data) if ids else cls.text(inner, data)
            if d is not None and cls.kind(d) == "identifier":
                return cls.text(d, data)
        return None

    @classmethod
    def extract_defs(cls, data: bytes, lang_key: str,
                     parser=None) -> Tuple[List[Tuple[str, Any]], bytes]:
        """[(name, def_node)] for every function/method definition."""
        parser = parser or cls.parser(lang_key)
        tree, data = cls.parse(parser, data)
        out: List[Tuple[str, Any]] = []
        for node in cls.walk(cls.root(tree)):
            if cls.kind(node) in DEF_NODE_TYPES:
                name = cls.def_name(node, data, lang_key)
                if name:
                    out.append((name, node))
        return out, data

    @classmethod
    def function_names(cls, func_code: str, lang_key: str) -> set:
        """Names of the definitions in a code snippet (a merged `function`
        column can contain several). Java bare methods parse only inside a
        class wrapper."""
        try:
            defs, _ = cls.extract_defs(func_code.encode(), lang_key)
            names = {nm for nm, _n in defs if nm}
            if not names and lang_key == "java":
                defs, _ = cls.extract_defs(
                    b"class _W{\n" + func_code.encode() + b"\n}", lang_key)
                names = {nm for nm, _n in defs if nm}
            return names
        except Exception:
            return set()

    @classmethod
    def build_def_index(cls, files: List[Path], repo_dir: Path,
                        lang_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """name -> [{file, function}] over the given source files."""
        parser = cls.parser(lang_key)
        index: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for fp in files:
            try:
                defs, data = cls.extract_defs(fp.read_bytes(), lang_key, parser)
            except Exception:
                continue
            try:
                rel = str(fp.relative_to(repo_dir))
            except Exception:
                rel = fp.name
            for name, node in defs:
                code = cls.text(node, data)
                if code:
                    index[name].append({"file": rel, "function": code})
        return index
