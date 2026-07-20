import anthropic
from google import genai
import tiktoken
from transformers import AutoTokenizer


class Evaluator:
    def __init__(self, model:str="gpt-5"):
        self.model = model
        
    def claude(self, prompt:str, model:str="claude-sonnet-4-5") -> int:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
        
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.count_tokens(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        return response.input_tokens
    
    def gemini(self, prompt:str, model:str="gemini-2.0-flash") -> int:
        from dotenv import load_dotenv
        import os, time
        load_dotenv()
        # On the throttled free tier, the count_tokens API competes with the
        # generate quota (5 RPM) and blocks; use a tiktoken estimate instead.
        if os.getenv("GEMINI_TOKENS_TIKTOKEN") == "1":
            return len(tiktoken.get_encoding("cl100k_base").encode(str(prompt)))
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        client = genai.Client(api_key=GEMINI_API_KEY)
        # count_tokens has a separate (higher) quota, but retry + fall back so a
        # transient rate limit never crashes the detection run.
        for attempt in range(4):
            try:
                response = client.models.count_tokens(model=model, contents=prompt)
                return int(getattr(response, "total_tokens", 0))
            except Exception as e:
                if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < 3:
                    time.sleep(15)
                    continue
                break
        return len(tiktoken.get_encoding("cl100k_base").encode(str(prompt)))
    
    def gpt(self, prompt:str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(prompt)))
    
    _TOK_CACHE = {}

    def transformer(self, prompt:str, model:str="qwen3-30b-instruct") -> int:
        # experiment panel (pair-pair-pair): map the clean served-model-id
        # (what run.py -m and vLLM --served-model-name use) to the HF repo
        # whose tokenizer to load. The NVIDIA hybrid serves both modes from the
        # same repo (two clean ids). A raw HF id ('/') is used directly;
        # anything unmapped falls back to a tiktoken estimate.
        models = {
            # Qwen pair (separate checkpoints)
            "qwen3-30b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "qwen3-30b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
            # Mistral pair (separate checkpoints)
            "mistral-small:24b": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "magistral-small:24b": "mistralai/Magistral-Small-2509",
            # NVIDIA pair (one hybrid, toggled by control token; same tokenizer)
            "nemotron-nano:12b": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "nemotron-nano-think:12b": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        }
        # sft adapters are served as "<base>-sft" (the vLLM lora-module id);
        # they share the base checkpoint's tokenizer.
        if model.endswith("-sft") and model[:-4] in models:
            models[model] = models[model[:-4]]
        # a raw HF repo id (contains '/') is a valid tokenizer source as-is
        if model not in models and "/" in model:
            models[model] = model
        # Cache tokenizers (avoid reloading per row) and fall back to a
        # tiktoken estimate if the HF tokenizer/config is incompatible
        # (e.g. Phi-3 'longrope' validation under newer transformers).
        try:
            tok = Evaluator._TOK_CACHE.get(model)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(models[model])
                Evaluator._TOK_CACHE[model] = tok
            return len(tok.encode(prompt))
        except Exception:
            return len(tiktoken.get_encoding("cl100k_base").encode(str(prompt)))
    
    def token_count(self, prompt:str) -> int:
        if self.model.startswith("claude"):
            return self.claude(prompt, model=self.model)
        if self.model.startswith("gemini"):
            return self.gemini(prompt, model=self.model)
        if self.model.startswith("gpt"):
            return self.gpt(prompt)
        return self.transformer(prompt, model=self.model)
        
