# Role

You are an experienced developer who knows the security vulnerability very well.

# Task

Predict Whether the C function below is vulnerable. Strictly return 1 for a vulnerable function and 0 for a non-vulnerable function without further explanation.

# Input

Function-level: Target function only

File-level: Entire file containing target function

Repository-level: Target function + file context + summarized caller/callee

# Output Format

## Vulnerable (0: no, 1: yes):
{{ "vulnerable": 0 or 1 (bool) }}
