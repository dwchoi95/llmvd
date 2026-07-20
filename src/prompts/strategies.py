"""Prompting-strategy registry for the scope x strategy experiment.

Three strategies, each an intervention of a different KIND:

  * zero — the baseline: the target (+ scope context) with a zero-shot
           instruction. Prompt folder: src/prompts/zero/.
  * rag  — a PROMPT-level intervention: prepend one retrieved vulnerable and
           one retrieved non-vulnerable example (GRACE-style). Prompt folder:
           src/prompts/rag/. Needs the example file (uses_examples=True).
  * sft  — a WEIGHT-level intervention: the SAME zero prompt, but served by a
           model fine-tuned on the training corpus. No prompt of its own —
           it reuses the zero folder; the difference is which model serves it.

`reasoning` (whether the served model emits a thinking trace before its
verdict) is a MODEL property, not a strategy property, so it lives on the
Detector, not here. Both direct-answer and thinking models run all three
strategies.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str          # results key: "zero" | "rag" | "sft" | "cot"
    prompt_dir: str    # which src/prompts/<dir>/ folder to render from
    uses_examples: bool  # rag injects retrieved vul/non-vul examples
    free_form: bool = False  # cot: no JSON-schema forcing; model reasons in-channel before the verdict


STRATEGIES: dict[str, Strategy] = {
    "zero": Strategy("zero", "zero", uses_examples=False),
    "rag":  Strategy("rag",  "rag",  uses_examples=True),
    # sft reuses the zero prompt; the intervention is the fine-tuned model
    "sft":  Strategy("sft",  "zero", uses_examples=False),
    # cot — a PROMPT-level reasoning control: same inputs, but the system
    # prompt asks for step-by-step reasoning before the final JSON verdict.
    # Free-form output (no schema forcing) so non-reasoning models get an
    # in-channel scratchpad; used to test whether "reasoning" is a model
    # property or merely permission to reason (threats: construct validity).
    "cot":  Strategy("cot",  "cot",  uses_examples=False, free_form=True),
}

DEFAULT_STRATEGIES = ["zero", "rag"]


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}"
        )
    return STRATEGIES[name]
