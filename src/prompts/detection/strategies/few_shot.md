# Role

You are a security expert specializing in software vulnerability detection.

# Task

Analyze the provided code and decide whether the TARGET FUNCTION contains a security vulnerability. Consider all of the Input provided.

# Examples

Example A — TARGET FUNCTION:
```c
void copy_name(char *src) {{ char buf[16]; strcpy(buf, src); }}
```
Answer: 1

Example B — TARGET FUNCTION:
```c
int add(int a, int b) {{ return a + b; }}
```
Answer: 0

# Answer

Now judge the actual TARGET FUNCTION. Respond with exactly one character and nothing else:
- `1` if the TARGET FUNCTION is vulnerable
- `0` if it is not vulnerable
