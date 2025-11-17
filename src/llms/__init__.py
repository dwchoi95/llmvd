import re
import typing as _t
from pydantic import BaseModel

class TextFormat(BaseModel):
    vulnerable: bool

__all__ = ["TextFormat", "GPT", "CLAUDE", "GEMINI", "OLLAMA"]

# 타입 체커용(런타임 임포트 안 함)
if _t.TYPE_CHECKING:
    from .gpt import GPT
    from .claude import CLAUDE
    from .gemini import GEMINI
    from .ollama import OLLAMA

# 지연 재노출로 순환 방지
def __getattr__(name: str):
    if name == "GPT":
        from .gpt import GPT
        return GPT
    if name == "CLAUDE":
        from .claude import CLAUDE
        return CLAUDE
    if name == "GEMINI":
        from .gemini import GEMINI
        return GEMINI
    if name == "OLLAMA":
        from .ollama import OLLAMA
        return OLLAMA
    raise AttributeError(f"module {__name__} has no attribute {name}")