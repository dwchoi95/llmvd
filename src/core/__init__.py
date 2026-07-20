"""Lazy re-exports so importing one component does not drag in the others'
heavy deps (Detector needs codebleu; SFT needs torch/peft). Mirrors the
lazy `__getattr__` pattern in `src/llms/__init__.py`."""

__all__ = ["Detector", "Evaluator", "SFT"]

_MODULES = {"Detector": "detector", "Evaluator": "evaluator", "SFT": "sft"}


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{module}", __name__), name)
