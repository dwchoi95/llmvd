from ollama import AsyncClient

from . import TextFormat

    
class LLAMA:
    def __init__(self, 
                 model:str="llama3:8b", 
                 temperature:float=0.0,
                 timeout:int=10):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")
        
        self.host = LOCAL_API_URL
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.client = AsyncClient(host=LOCAL_API_URL)

    async def run(self, system:str, user:str, max_retry:int=1) -> str| None:
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={
                    "temperature": self.temperature,
                    "timeout": self.timeout
                },
                format=TextFormat.model_json_schema()
            )
            res = TextFormat.model_validate_json(response.message.content)
            vulnerable = res.vulnerable
            return vulnerable
        except Exception as e:
            print(e)
            import time
            if max_retry > 0:
                time.sleep(5)
                return await self.run(system, user, max_retry-1)
        return None
        