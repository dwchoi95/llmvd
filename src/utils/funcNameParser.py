from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

QUERIES = {
    "python": r"""
(function_definition name: (identifier) @name)
(decorated_definition
  (function_definition name: (identifier) @name))
(class_definition
  name: (identifier) @name)
    """,
    "java": r"""
(method_declaration name: (identifier) @name)
(constructor_declaration name: (identifier) @name)
    """,
    "c": r"""
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name))

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name)))

(function_definition
  declarator: (pointer_declarator
    declarator: (pointer_declarator
      declarator: (function_declarator
        declarator: (identifier) @name))))

(function_definition
  declarator: (parenthesized_declarator
    (identifier) @name))
    """,
    "cpp": r"""
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name))

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (identifier) @name)))

(function_definition
  declarator: (pointer_declarator
    declarator: (pointer_declarator
      declarator: (function_declarator
        declarator: (identifier) @name))))

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator
      declarator: (qualified_identifier
        name: (identifier) @name))))

(function_definition
  declarator: (parenthesized_declarator
    (identifier) @name))

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name)))
    """,
}


class FuncNameParser:
    @staticmethod
    def run(code: str, lang: str) -> str:
        key = {"python":"python","py":"python",
            "java":"java",
            "c":"c",
            "c++":"cpp","cpp":"cpp","cc":"cpp","cxx":"cpp"}[lang.lower()]
        parser = get_parser(key)
        tree = parser.parse(code.encode("utf-8"))
        language = get_language(key)
        
        query = Query(language, QUERIES[key])
        cursor = QueryCursor(query)
        
        caps = cursor.captures(tree.root_node)
        
        function_names = []
        for node in caps.get("name", []):
            func_name = code[node.start_byte:node.end_byte]
            function_names.append(func_name)
        
        lines = code.splitlines()
        for line in lines:
          for func_name in function_names:
            if func_name in line:
              return func_name.strip()
            
        return None