"""Anthropic LLM provider with retry, response caching, and cost recording.

This is the only place in the codebase that talks to the Anthropic SDK.
Every call is:
  1. checked against the SQLite response cache  (skip cost on hit)
  2. retried with exponential backoff on rate-limit / overload / timeout
  3. logged with token usage + USD cost
  4. wrapped in a tracing span carrying model, tokens, and cache status
"""
from __future__ import annotations
import base64
import random
import time
from anthropic import Anthropic
from anthropic._exceptions import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from config import settings
from errors import LLMError, LLMRateLimited, LLMTimeout
from logging_setup import get_logger
from obs import cache, costs
from obs.tracer import span
from providers.base import LLMProvider, LLMResponse


log = get_logger(__name__)

MAX_RETRIES = 4
BASE_DELAY = 0.5


class AnthropicLLM(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or settings.anthropic_api_key
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._client = Anthropic(api_key=key)
        self._model = model or settings.anthropic_model

    # ---- public API -----------------------------------------------------
    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> LLMResponse:
        return self._call(system=system, messages=messages, max_tokens=max_tokens)

    def complete_vision(
        self, system: str, image_png: bytes, instruction: str, max_tokens: int = 4096
    ) -> LLMResponse:
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return self._call(system=system, messages=messages, max_tokens=max_tokens)

    # ---- internals ------------------------------------------------------
    def _call(self, system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        with span("llm.call", model=self._model, max_tokens=max_tokens) as s:
            key = cache.make_key(self._model, system, messages, max_tokens)
            hit = cache.get(key)
            if hit is not None:
                s.set_many(cached=True, input_tokens=hit.input_tokens, output_tokens=hit.output_tokens)
                return LLMResponse(
                    text=hit.response_text,
                    input_tokens=hit.input_tokens,
                    output_tokens=hit.output_tokens,
                    model=self._model,
                    cached=True,
                )

            delay = BASE_DELAY
            last_err: Exception | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self._client.messages.create(
                        model=self._model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                    )
                    text = "".join(
                        b.text for b in resp.content if getattr(b, "type", None) == "text"
                    ) or (resp.content[0].text if resp.content else "")
                    in_t = getattr(resp.usage, "input_tokens", 0)
                    out_t = getattr(resp.usage, "output_tokens", 0)

                    cache.put(key, self._model, text, in_t, out_t)
                    usd = costs.record("anthropic", self._model, in_t, out_t)
                    s.set_many(input_tokens=in_t, output_tokens=out_t, usd=usd, attempts=attempt + 1)
                    return LLMResponse(
                        text=text, input_tokens=in_t, output_tokens=out_t,
                        model=self._model, cached=False,
                    )
                except RateLimitError as e:
                    last_err = e
                    log.warning("anthropic rate-limited", extra={"ctx_attempt": attempt + 1})
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    last_err = e
                    log.warning("anthropic transient error", extra={
                        "ctx_attempt": attempt + 1, "ctx_error": str(e),
                    })
                except Exception as e:
                    s.status = "error"
                    s.error = f"{type(e).__name__}: {e}"
                    raise LLMError(str(e)) from e

                time.sleep(delay + random.random() * 0.2)
                delay *= 2

            s.status = "error"
            s.error = f"exhausted_retries: {last_err}"
            if isinstance(last_err, RateLimitError):
                raise LLMRateLimited(str(last_err))
            if isinstance(last_err, APITimeoutError):
                raise LLMTimeout(str(last_err))
            raise LLMError(f"exhausted retries: {last_err}")
