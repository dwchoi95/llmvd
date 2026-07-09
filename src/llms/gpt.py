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
        # reasoning effort for reasoning models (gpt-5 family); low keeps cost down
        self.reasoning_effort = os.getenv("GPT_REASONING_EFFORT")

    async def run(self, system:str, user:str, retry:int=3) -> str | None:
        try:
            kwargs = dict(
                model=self.model,
                temperature=self.temperature,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                text_format=TextFormat,
            )
            if self.reasoning_effort and self.model.startswith("gpt-5"):
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            response = await self.client.responses.parse(**kwargs)
            model = getattr(response, "output_parsed", None)
            if model is not None:
                vulnerable = getattr(model, "vulnerable", None)
                return vulnerable
        except Exception as e:
            print(e)
            # import time
            # time.sleep(60)
            # return await self.run(system, user, retry - 1)
            pass
        return None
