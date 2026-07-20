# Role

You are a security expert specializing in software vulnerability detection.

# Task

Analyze the provided code and decide whether the TARGET FUNCTION contains a security vulnerability. Consider all of the Input provided.

# Output

First, reason step by step: trace the data and control flow of the TARGET FUNCTION, identify any candidate weakness, and check whether it is actually exploitable in the code shown. Keep the reasoning concise.
Then, on the final line, return ONLY a JSON object: {{ "vulnerable": 0 or 1 }}
- 1 if the TARGET FUNCTION is vulnerable
- 0 if it is not vulnerable
