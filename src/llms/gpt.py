from openai import AsyncOpenAI, OpenAIError, BadRequestError, APITimeoutError
from . import TextFormat

class GPT:
    def __init__(self,
                 model:str="gpt-3.5-turbo",
                 temperature:float=0.0,
                 timeout:int=10):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    async def run(self, system:str, user:str, max_retry:int=3) -> str | None:
        try:
            try:
                response = await self.client.responses.parse(
                    model=self.model,
                    timeout=self.timeout,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    text_format=TextFormat,
                )
            except OpenAIError:
                response = await self.client.responses.create(
                    model=self.model,
                    timeout=self.timeout,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                )
            except (APITimeoutError) as e:
                import time
                if max_retry > 0:
                    time.sleep(5)
                    return await self.run(system, user, max_retry-1)
                return None

            model = getattr(response, "output_parsed", None)
            if model is not None:
                vulnerable = getattr(model, "vulnerable", None)
                return vulnerable

            vulnerable = getattr(response, "output_text", None)
            return vulnerable
        except (APITimeoutError) as e:
            import time
            if max_retry > 0:
                time.sleep(5)
                return await self.run(system, user, max_retry-1)
            return None
        except Exception as e:
            print(e)
        return None
