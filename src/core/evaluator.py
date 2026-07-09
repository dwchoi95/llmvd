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

    def transformer(self, prompt:str, model:str="llama3.1:8b") -> int:
        models = {
            "llama3.1:8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "deepseek-coder-v2:16b": "deepseek-ai/DeepSeek-Coder-V2-Base",
            "phi3:14b": "microsoft/Phi-3-medium-128k-instruct",
            "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
            "qwen3-coder:30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "deepseek-coder-v2:16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        }
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
        
