import anthropic
from anthropic.types import ToolUseBlock

from . import TextFormat


class CLAUDE:
    def __init__(self,
                 model:str="claude-opus-4-1",
                 temperature:float=0.0):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
        
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        self.async_client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)
        self.model = model
        self.temperature = temperature
        self._tool_name = "structured_output"
        self._schema = {
            "type": "object",
            "properties": {
                "vulnerable": {"type": "boolean"}
            },
            "required": ["vulnerable"],
            "additionalProperties": False
        }
    
    def _extract_response(self, response):
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == self._tool_name:
                payload = block.input
                if isinstance(payload, dict) and "vulnerable" in payload:
                    return payload.get("vulnerable")
        response_texts = [block.text for block in response.content if hasattr(block, 'text')]
        vulnerable = " ".join(response_texts)
        return vulnerable

    async def run(self, system:str, user:str, max_retries:int=1) -> str | None:
        try:
            response = await self.async_client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=1000,
                system=system,
                messages=[
                    {"role": "user", "content": user}
                ],
                tools=[
                    {
                        "name": self._tool_name,
                        "description": "Return the final answer under the fixed key.",
                        "input_schema": self._schema
                    }
                ],
                tool_choice={"type": "tool", "name": self._tool_name},
                extra_headers={"anthropic-beta": "tools-2024-04-04"}
            )
            return self._extract_response(response)
        except Exception as e:
            # sleep and retry
            if max_retries > 0:
                import time
                time.sleep(10)
                return await self.run(system, user, max_retries - 1)
            else:
                print(e)
        return None