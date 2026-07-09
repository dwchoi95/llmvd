"""Prompting-strategy registry for the scope x prompting x tier experiment.

Each strategy pairs a system-prompt template with the decoding budget it needs.
Reasoning strategies (CoT, Think&Verify) emit their verdict on a final
`FINAL: 1/0` line after free-text reasoning, so they need a large token budget;
direct strategies (zero-shot, few-shot) emit a single `1`/`0` token.

The score-capture client (OLLAMA.run_scored) locates the decision token by
scanning for the LAST positive/negative answer token, which works for both the
single-token and the `FINAL: x` forms.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    system_file: str
    max_tokens: int
    reasoning: bool  # True => free-text reasoning precedes the verdict


_DIR = "src/prompts/detection/strategies"

STRATEGIES: dict[str, Strategy] = {
    # direct: schema-forced JSON verdict {"vulnerable": 0|1} (needs ~6-10 tokens)
    "zero_shot":    Strategy("zero_shot",    f"{_DIR}/zero_shot.md",    max_tokens=16,  reasoning=False),
    "few_shot":     Strategy("few_shot",     f"{_DIR}/few_shot.md",     max_tokens=16,  reasoning=False),
    # reasoning, verdict on final line
    "cot":          Strategy("cot",          f"{_DIR}/cot.md",          max_tokens=768, reasoning=True),
    "think_verify": Strategy("think_verify", f"{_DIR}/think_verify.md", max_tokens=1024, reasoning=True),
}

DEFAULT_STRATEGIES = ["zero_shot", "few_shot", "cot", "think_verify"]


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}"
        )
    return STRATEGIES[name]
