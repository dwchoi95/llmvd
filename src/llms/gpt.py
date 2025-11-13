from openai import AsyncOpenAI, OpenAIError, BadRequestError, APITimeoutError
from . import TextFormat

class GPT:
    def __init__(self,
                 model:str="gpt-5",
                 temperature:float=1.0):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.temperature = temperature

    async def run(self, system:str, user:str) -> str | None:
        try:
            response = await self.client.responses.parse(
                model=self.model,
                temperature=self.temperature,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                text_format=TextFormat,
            )
            model = getattr(response, "output_parsed", None)
            if model is not None:
                vulnerable = getattr(model, "vulnerable", None)
                return vulnerable
        except Exception as e:
            print(e)
            pass
        return None
