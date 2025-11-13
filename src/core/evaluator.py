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
        import os
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.count_tokens(
            model=model, contents=prompt
        )
        return int(getattr(response, "total_tokens", 0))
    
    def gpt(self, prompt:str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(prompt)))
    
    def transformer(self, prompt:str, model:str="llama3.1:8b") -> int:
        models = {
            "llama3.1:8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "deepseek-coder-v2:16b": "deepseek-ai/DeepSeek-Coder-V2-Base",
            "phi3:14b": "microsoft/Phi-3-medium-128k-instruct",
            "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
            "qwen3-coder:30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        }
        tokenizer = AutoTokenizer.from_pretrained(models[model])
        return len(tokenizer.encode(prompt))
    
    def token_count(self, prompt:str) -> int:
        if self.model.startswith("claude"):
            return self.claude(prompt, model=self.model)
        if self.model.startswith("gemini"):
            return self.gemini(prompt, model=self.model)
        if self.model.startswith("gpt"):
            return self.gpt(prompt)
        return self.transformer(prompt, model=self.model)
        
