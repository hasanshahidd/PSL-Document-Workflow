"""SQLite-backed cache for LLM responses.

Keyed by SHA256 of (model, system, user_messages, max_tokens). A cache hit
returns the recorded response text and token counts and skips both the
network call and the cost ledger. The eval orchestrator can therefore
re-run cheaply.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from storage.db import get_conn


@dataclass
class CacheHit:
    response_text: str
    input_tokens: int
    output_tokens: int


def make_key(model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps(
        {"m": model, "s": system, "msgs": messages, "mt": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> CacheHit | None:
    row = get_conn().execute(
        "SELECT response_text, input_tokens, output_tokens "
        "FROM llm_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    return CacheHit(
        response_text=row["response_text"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
    )


def put(key: str, model: str, response_text: str, input_tokens: int, output_tokens: int) -> None:
    get_conn().execute(
        "INSERT OR REPLACE INTO llm_cache(cache_key, model, response_text, "
        "input_tokens, output_tokens) VALUES(?,?,?,?,?)",
        (key, model, response_text, input_tokens, output_tokens),
    )
    get_conn().commit()
