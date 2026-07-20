import os
import re
import math

import httpx
from ollama import AsyncClient

from . import TextFormat


# Tokens that vote "vulnerable" (positive) vs "not vulnerable" (negative) when
# reading the decision-token logprob distribution. Compared case-insensitively
# against the stripped token text.
_POS_TOKENS = {"1", "yes", "true", "vulnerable", "vuln", "insecure", "unsafe"}
_NEG_TOKENS = {"0", "no", "false", "safe", "secure", "not", "none", "clean"}


def _norm(tok: str) -> str:
    return tok.strip().strip('"').strip("'").lower()


class OLLAMA:
    def __init__(self,
                 model: str = "llama3.1:8b",
                 temperature: float | None = None):
        from dotenv import load_dotenv
        load_dotenv()
        LOCAL_API_URL = os.getenv("LOCAL_API_URL")

        self.host = (LOCAL_API_URL or "").rstrip("/")
        self.model = model
        # None => do not send temperature/top_p; the server applies the served
        # model's own generation_config defaults (reproducibility: each model
        # runs at its shipped sampling settings). Our score comes from the
        # verdict-token LOGPROBS, which are pre-sampling and hence unaffected by
        # temperature, so leaving it at the model default does not bias scores.
        self.temperature = temperature
        self.client = AsyncClient(host=LOCAL_API_URL)
        # OpenAI-compatible endpoint (Ollama serves it at /v1) -- the only path
        # that returns per-token logprobs, which we need for a P(vulnerable)
        # score and hence a threshold-free ROC/AUC decomposition.
        self.openai_url = f"{self.host}/v1/chat/completions"

    # ------------------------------------------------------------------ #
    # label-only path (unchanged): structured boolean via JSON schema.
    # Kept for backward-compatible reproduction of existing result files.
    # ------------------------------------------------------------------ #
    async def run(self, system: str, user: str) -> "str | bool | None":
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.temperature},
                format=TextFormat.model_json_schema(),
            )
            res = TextFormat.model_validate_json(response.message.content)
            return res.vulnerable
        except Exception as e:
            print(e)
        return None

    # ------------------------------------------------------------------ #
    # scored path: returns (label, score, raw_text).
    #   label -> bool | None   (None when neither answer token is observed)
    #   score -> P(vulnerable) in [0,1] | None
    # For zero-shot the answer is a single token (max_tokens small); for
    # reasoning prompts (CoT/verify) pass a larger max_tokens and the decision
    # token is located by scanning for the LAST positive/negative answer token.
    # ------------------------------------------------------------------ #
    # forces a valid {"vulnerable": 0|1} for direct strategies so large inputs
    # cannot make the model ramble past the instruction (a 40-66% failure mode
    # on multi-file inputs otherwise); logprobs are still returned for the digit
    _VULN_SCHEMA = {
        "type": "object",
        "properties": {"vulnerable": {"type": "integer", "enum": [0, 1]}},
        "required": ["vulnerable"],
    }

    # generous transport timeout: with the client keeping the server's queue
    # full, a request's clock includes queue wait + 32k-token prompt
    # processing + up to 1024 generated tokens. 300s was observed to mass-
    # expire reasoning calls whenever the server got contended.
    async def run_scored(self, system: str, user: str,
                         max_tokens: int = 16,
                         reasoning: bool = False,
                         top_logprobs: int = 20,
                         timeout: float = 900.0):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        # only override temperature if explicitly set; otherwise the server
        # uses the served model's generation_config default (reproducibility)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        # direct strategies: constrain output to JSON so it never rambles.
        # reasoning strategies stay free-form (they must reason before verdict).
        if not reasoning:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": self._VULN_SCHEMA},
            }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.openai_url,
                    json=payload,
                    headers={"Authorization": "Bearer ollama"},
                )
                resp.raise_for_status()
                data = resp.json()
            choice = data["choices"][0]
            raw = choice["message"]["content"]
            label, score = self._decide(choice.get("logprobs"), raw, reasoning)
            return label, score, raw
        except Exception as e:
            print(e)
        return None, None, None

    @staticmethod
    def _mass(alts, vocab) -> float:
        return sum(math.exp(a["logprob"]) for a in alts
                   if _norm(a["token"]) in vocab)

    # a position only counts as a real verdict if at least this much of its
    # probability mass sits on answer tokens -- otherwise the model is not
    # actually answering here (e.g. it started prose), and a score read off the
    # deep tail of the distribution would be a fabricated verdict.
    MIN_ANSWER_MASS = 0.5

    @classmethod
    def _score_at(cls, tokinfo) -> "float | None":
        """P(vulnerable) at one decoding position from its logprob distribution.

        We force a '1'/'0' answer, so the decision variable is P('1') vs P('0');
        fall back to the broader semantic vocabulary only if neither canonical
        digit is present. Returns None when the position is not a genuine verdict
        (answer mass below MIN_ANSWER_MASS).
        """
        alts = tokinfo.get("top_logprobs") or [
            {"token": tokinfo["token"], "logprob": tokinfo["logprob"]}
        ]
        p1, p0 = cls._mass(alts, {"1"}), cls._mass(alts, {"0"})
        if p1 + p0 >= cls.MIN_ANSWER_MASS:
            return p1 / (p1 + p0)
        p_pos, p_neg = cls._mass(alts, _POS_TOKENS), cls._mass(alts, _NEG_TOKENS)
        if p_pos + p_neg >= cls.MIN_ANSWER_MASS:
            return p_pos / (p_pos + p_neg)
        return None

    @classmethod
    def _pick_score(cls, positions) -> "float | None":
        """First position (in the given order) that is a genuine verdict, i.e.
        whose answer-mass clears MIN_ANSWER_MASS. Skips positions like ':' or ' '
        that merely carry a digit in the deep tail of their distribution."""
        for t in positions:
            s = cls._score_at(t)
            if s is not None:
                return s
        return None

    @classmethod
    def _decide(cls, logprobs, raw, reasoning=False):
        """Return (label, score).

        Both come from the verdict position's logprob DISTRIBUTION, not the
        sampled token: the /v1 endpoint does not always honour temperature=0
        (it can emit a lower-probability token), so the sampled digit is
        neither deterministic nor reliable, whereas P('1') vs P('0') is.
        label = (score >= 0.5) keeps the hard label consistent with the score
        and with the ROC operating point.

        Reasoning models emit a thinking trace and then the verdict, so the
        answer is AFTER the last `</think>` (or `final`) marker -> scan forward
        from there for the first committed digit; fall back to the last
        committed digit anywhere. Direct models emit the schema-forced JSON
        verdict up front -> scan the first few positions ({"vulnerable": <d>).
        """
        content = (logprobs or {}).get("content") if logprobs else None
        if not content:
            return cls._label_from_text(raw), None

        if reasoning:
            # panel reasoning models close their thinking trace with different
            # delimiters: Qwen/R1 `</think>`, Mistral Magistral `[/THINK]`,
            # NVIDIA Nemotron `</think>` or channel markers; some just reason in
            # prose. Take the LAST end-of-thinking marker (or `FINAL:`) and scan
            # forward for the first committed verdict digit; if there is no
            # marker, the verdict is simply the last committed digit.
            marker_idx = -1
            for i, t in enumerate(content):
                tok = _norm(t["token"])
                is_think_end = (("think" in tok and (">" in tok or "]" in tok))
                                or "channel" in tok or tok.startswith("final"))
                if is_think_end:
                    marker_idx = i
            if marker_idx >= 0:
                score = cls._pick_score(content[marker_idx + 1:])
            else:  # no marker -> the final answer is the last committed digit
                score = cls._pick_score(list(reversed(content)))
        else:
            # direct strategies emit the verdict up front; for schema-forced
            # JSON the digit sits a few tokens in ({"vulnerable": <digit>).
            score = cls._pick_score(content[:16])

        if score is None:
            return cls._label_from_text(raw), None
        return (score >= 0.5), score

    @staticmethod
    def _label_from_text(raw) -> "bool | None":
        """Fallback verdict when logprobs are unavailable: parse raw text,
        preferring the token after a FINAL marker, else the first answer word."""
        s = (raw or "").lower()
        if "final" in s:
            s = s[s.rfind("final"):]
        for tok in re.split(r"[^a-z0-9]+", s):
            if tok in _POS_TOKENS:
                return True
            if tok in _NEG_TOKENS:
                return False
        return None
