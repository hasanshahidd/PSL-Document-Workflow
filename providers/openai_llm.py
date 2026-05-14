"""OpenAI LLM provider — same shape as the Anthropic one.

This file is the only place that imports `openai`. The rest of the codebase
talks to the LLMProvider ABC. Pick this one by setting `LLM_PROVIDER=openai`
(default when OPENAI_API_KEY is set) — see providers/registry.py.

Same goodies as the Anthropic provider:
  - sqlite-backed response cache (skips network + cost on hit)
  - retry with exponential backoff + jitter on rate-limit / transient errors
  - per-call cost recording tagged with the active trace/span id
  - tracing span around every call
"""
from __future__ import annotations
import base64
import random
import time
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
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


class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or settings.openai_api_key
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)
        self._model = model or settings.openai_model

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
                    {"type": "text", "text": instruction},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]
        return self._call(system=system, messages=messages, max_tokens=max_tokens)

    # ---- internals ------------------------------------------------------
    def _call(self, system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        with span("llm.call", provider="openai", model=self._model, max_tokens=max_tokens) as s:
            key = cache.make_key(self._model, system, messages, max_tokens)
            hit = cache.get(key)
            if hit is not None:
                s.set_many(cached=True,
                           input_tokens=hit.input_tokens,
                           output_tokens=hit.output_tokens)
                return LLMResponse(
                    text=hit.response_text,
                    input_tokens=hit.input_tokens,
                    output_tokens=hit.output_tokens,
                    model=self._model,
                    cached=True,
                )

            # OpenAI puts the system prompt inside the messages list.
            full_messages = [{"role": "system", "content": system}] + messages

            delay = BASE_DELAY
            last_err: Exception | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=full_messages,
                        max_tokens=max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    in_t = getattr(resp.usage, "prompt_tokens", 0) or 0
                    out_t = getattr(resp.usage, "completion_tokens", 0) or 0

                    cache.put(key, self._model, text, in_t, out_t)
                    usd = costs.record("openai", self._model, in_t, out_t)
                    s.set_many(input_tokens=in_t, output_tokens=out_t,
                               usd=usd, attempts=attempt + 1)
                    return LLMResponse(
                        text=text, input_tokens=in_t, output_tokens=out_t,
                        model=self._model, cached=False,
                    )
                except RateLimitError as e:
                    last_err = e
                    log.warning("openai rate-limited",
                                extra={"ctx_attempt": attempt + 1})
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    last_err = e
                    log.warning("openai transient error", extra={
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
