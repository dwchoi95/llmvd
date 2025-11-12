"""Tree edit distance utilities based on tree-sitter ASTs."""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language, get_parser
from typing import Optional
from zss import Node as ZssNode
from zss import simple_distance


COMMENT_NODE_TYPES = {"comment", "block_comment", "line_comment"}


class _ZssNode(ZssNode):
    """A thin wrapper over :class:`zss.Node` for type clarity."""

    def __init__(self, label: str) -> None:
        super().__init__(label)


class TED:
    """Compute tree edit distance (TED) between two C/C++ snippets.

    The code is parsed with tree-sitter and converted into a simplified AST where
    comment nodes are discarded. The resulting trees are compared using the
    Zhang–Shasha algorithm provided by :mod:`zss`.
    """

    def __init__(self, language: str = "c") -> None:
        language = language.lower()
        if language == "c":
            lang = get_language("c")
        elif language == "cpp" or language == "c++":
            lang = get_language("cpp")
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        self.lang = Language(lang)
        self._parser = get_parser(language)

    def _to_zss(self, node, source: bytes) -> Optional[_ZssNode]:
        if node.type in COMMENT_NODE_TYPES:
            return None

        label = node.type
        if node.child_count == 0:
            text = source[node.start_byte : node.end_byte].decode("utf-8", "ignore").strip()
            if text:
                label = f"{label}:{text}"

        zss_node = _ZssNode(label)
        for child in node.children:
            converted = self._to_zss(child, source)
            if converted is not None:
                zss_node.addkid(converted)

        return zss_node

    def run(self, code_a: str, code_b: str) -> int:
        """Return the tree edit distance between *code_a* and *code_b*.

        Comments are ignored. If either code cannot be parsed, a ``ValueError`` is
        raised.
        """

        source_a = code_a.encode("utf-8")
        source_b = code_b.encode("utf-8")

        tree_a = self._parser.parse(source_a)
        tree_b = self._parser.parse(source_b)

        root_a = self._to_zss(tree_a.root_node, source_a)
        root_b = self._to_zss(tree_b.root_node, source_b)

        if root_a is None or root_b is None:
            raise ValueError("Parsed trees do not contain comparable nodes.")

        return simple_distance(root_a, root_b)

