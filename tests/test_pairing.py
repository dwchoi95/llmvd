"""Pairing checks for Dataset._target_pairs / _primary_name.
Run: ./env/bin/python -m tests.test_pairing"""
from src.utils.dataset import Dataset as D

foo_b = "int foo(int x) {\n  return x;\n}"
foo_a = "int foo(int x) {\n  if (x < 0) return 0;\n  return x;\n}"
bar = "int bar(void) {\n  return 1;\n}"

# after-list is NOT position-aligned with before-list: pair by name
detail = {"function_before": [{"function": bar, "target": 0}, {"function": foo_b, "target": 1}],
          "function_after": [{"function": foo_a, "target": 0}, {"function": bar, "target": 0}]}
[(b, a)] = D._target_pairs(detail, "c")
assert (b["function"], a["function"]) == (foo_b, foo_a)

# unchanged target (or whitespace-only change) is not a patched pair
detail["function_after"] = [{"function": foo_b.replace("  ", "\t"), "target": 0}]
assert D._target_pairs(detail, "c") == []

# non-target functions never become the vulnerable member
detail["function_before"] = [{"function": bar, "target": 0}]
assert D._target_pairs(detail, "c") == []

# class snippets compare by class name, not by a shared __init__/first method
assert D._primary_name("class A(B):\n    def __init__(self):\n        pass\n", "python") == "A"
assert D._primary_name("public class C extends D {\n  void m() {}\n}", "java") == "C"
assert D._primary_name("MONGO_EXPORT int bson_size( const bson *b ) {\n return 1; }", "c") == "bson_size"
print("ok")
