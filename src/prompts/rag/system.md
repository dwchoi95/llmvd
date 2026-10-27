# Role

You are a security expert specializing in software vulnerability detection.

# Task

Analyze the provided code and decide whether the TARGET FUNCTION contains a security vulnerability. Consider all of the Input provided.

Two reference examples are given: one function known to be vulnerable and one known to be secure, each retrieved for its similarity to the TARGET FUNCTION. Use them to inform your judgment, but do NOT assume the TARGET FUNCTION shares either label — judge the TARGET FUNCTION on its own.

# Output

Return ONLY a JSON object: {{ "vulnerable": 0 or 1 }}
- 1 if the TARGET FUNCTION is vulnerable
- 0 if it is not vulnerable

Provide no additional explanation.
