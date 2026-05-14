"""Token + USD cost ledger.

Prices are configurable per model. Every LLM call records an entry tagged
with the current trace_id and span_id, so cost can be rolled up per draft,
per session, or per provider.
"""
from __future__ import annotations
from logging_setup import current_span_id, current_trace_id
from storage.db import get_conn


# USD per 1M tokens. Update from the model card when needed.
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "default": (3.00, 15.00),
}


def usd_for(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICING.get(model, PRICING["default"])
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0


def record(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    usd = usd_for(model, input_tokens, output_tokens)
    get_conn().execute(
        "INSERT INTO cost_ledger(trace_id, span_id, provider, model, "
        "input_tokens, output_tokens, usd) VALUES(?,?,?,?,?,?,?)",
        (current_trace_id(), current_span_id(), provider, model,
         input_tokens, output_tokens, usd),
    )
    get_conn().commit()
    return usd


def totals_for_trace(trace_id: str) -> dict:
    row = get_conn().execute(
        "SELECT COUNT(*) AS calls, SUM(input_tokens) AS in_t, "
        "SUM(output_tokens) AS out_t, SUM(usd) AS usd "
        "FROM cost_ledger WHERE trace_id = ?",
        (trace_id,),
    ).fetchone()
    return {
        "calls": row["calls"] or 0,
        "input_tokens": row["in_t"] or 0,
        "output_tokens": row["out_t"] or 0,
        "usd": round(row["usd"] or 0.0, 6),
    }


def grand_total() -> dict:
    row = get_conn().execute(
        "SELECT COUNT(*) AS calls, SUM(input_tokens) AS in_t, "
        "SUM(output_tokens) AS out_t, SUM(usd) AS usd FROM cost_ledger"
    ).fetchone()
    return {
        "calls": row["calls"] or 0,
        "input_tokens": row["in_t"] or 0,
        "output_tokens": row["out_t"] or 0,
        "usd": round(row["usd"] or 0.0, 6),
    }
