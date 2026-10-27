import asyncio
import os
import re

from google import genai
from google.genai.types import GenerateContentConfig

from . import TextFormat


class GEMINI:
    def __init__(self,
                 model:str="gemini-2.0-flash",
                 temperature:float=0.0):
        from dotenv import load_dotenv
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.async_client = self.client.aio
        self.model = model
        self.temperature = temperature
        # Free-tier throttling: minimum seconds between request starts.
        # Set GEMINI_MIN_INTERVAL=13 for the 5 req/min free tier; 0 disables it.
        self._min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", "0"))
        self._max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "8"))
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def _throttle(self):
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            wait = self._min_interval - (loop.time() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()

    @staticmethod
    def _retry_delay(msg: str, default: float = 15.0) -> float:
        m = re.search(r"retry in ([0-9.]+)s", msg) or re.search(r"retryDelay'?: ?'?([0-9.]+)s", msg)
        if m:
            try:
                return min(float(m.group(1)) + 1.0, 60.0)
            except Exception:
                pass
        return default

    async def run(self, system:str, user:str):
        for attempt in range(self._max_retries + 1):
            await self._throttle()
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
                    return getattr(model, "vulnerable", None)
                return None
            except Exception as e:
                msg = str(e)
                rate_limited = "RESOURCE_EXHAUSTED" in msg or "429" in msg
                overloaded = "UNAVAILABLE" in msg or "503" in msg or "overloaded" in msg
                # 429 (rate limit) -> wait the full hint; the throttle keeps us under it.
                # 503 (overload) -> only a couple of short retries, then give up fast and
                # let the resumable re-run fill the gap (avoids multi-minute blocking).
                # Give up quickly so a resumable pass stays bounded even when the
                # daily free quota is exhausted; the loop retries nulls later.
                if rate_limited and attempt < 2:
                    await asyncio.sleep(min(self._retry_delay(msg), 20))
                    continue
                if overloaded and attempt < 2:
                    await asyncio.sleep(min(3 * (2 ** attempt), 10))
                    continue
                print(e)
                return None
        return None
