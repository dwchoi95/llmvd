from google import genai
from google.genai.types import GenerateContentConfig

from . import TextFormat

    
class GEMINI:
    def __init__(self,
                 model:str="gemini-2.0-flash",
                 temperature:float=0.0):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.async_client = self.client.aio
        self.model = model
        self.temperature = temperature
    
    async def run(self, system:str, user:str):
        try:
            response = await self.async_client.models.generate_content(
                model=self.model,
                contents=user,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TextFormat,
                    system_instruction=system,
                    temperature=self.temperature
                ),
            )
            model = getattr(response, "parsed", None)
            if model is not None:
                vulnerable = getattr(model, "vulnerable", None)
                return vulnerable
        except Exception as e:
            print(e)
            pass
        return None