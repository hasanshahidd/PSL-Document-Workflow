"""Hierarchical tracing for the whole pipeline.

A `Span` is one logical operation (e.g. 'plan_queries', 'retrieve',
'llm_call'). Spans nest: opening one inside another sets its parent to the
currently active span. Every span is flushed to the `traces` SQLite table
when it closes, so a draft's entire request tree can be reconstructed by
filtering on its trace_id.

Usage:

    with span("generate_case_fact_summary", doc_ids=doc_ids) as s:
        ...
        s.set("evidence_count", len(evidence))

The `traced(name)` decorator gives the same behaviour for functions.
"""
from __future__ import annotations
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Iterator
from logging_setup import set_span_id, set_trace_id, current_trace_id
from storage.db import get_conn


_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    started_at: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_many(self, **kwargs: Any) -> None:
        self.attributes.update(kwargs)


def _flush(span: Span) -> None:
    duration_ms = (time.time() - span.started_at) * 1000.0
    try:
        get_conn().execute(
            "INSERT OR REPLACE INTO traces(span_id, trace_id, parent_span_id, name, "
            "started_at, duration_ms, status, attributes_json, error) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.name,
                datetime.fromtimestamp(span.started_at, tz=timezone.utc).isoformat(),
                duration_ms,
                span.status,
                json.dumps(span.attributes, default=str),
                span.error,
            ),
        )
        get_conn().commit()
    except Exception:
        # Tracing must never break the pipeline.
        pass


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    parent = _current_span.get()
    trace_id = parent.trace_id if parent else (current_trace_id() or uuid.uuid4().hex)
    new_span = Span(
        span_id=uuid.uuid4().hex,
        trace_id=trace_id,
        parent_span_id=parent.span_id if parent else None,
        name=name,
        attributes=dict(attrs),
    )
    token = _current_span.set(new_span)
    set_trace_id(trace_id)
    set_span_id(new_span.span_id)
    try:
        yield new_span
    except Exception as e:
        new_span.status = "error"
        new_span.error = f"{type(e).__name__}: {e}"
        _flush(new_span)
        raise
    else:
        _flush(new_span)
    finally:
        _current_span.reset(token)
        set_span_id(parent.span_id if parent else None)
        if parent is None:
            set_trace_id(None)


def traced(name: str | None = None) -> Callable:
    """Decorator form. Uses the function's qualified name when `name` is None."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name or fn.__qualname__):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def start_trace(trace_id: str | None = None) -> str:
    """Begin a top-level trace_id without opening a span yet."""
    tid = trace_id or uuid.uuid4().hex
    set_trace_id(tid)
    return tid


def current_span() -> Span | None:
    return _current_span.get()


def trace_tree(trace_id: str) -> list[dict]:
    """Return all spans for a trace ordered by start time."""
    rows = get_conn().execute(
        "SELECT span_id, trace_id, parent_span_id, name, started_at, duration_ms, "
        "status, attributes_json, error "
        "FROM traces WHERE trace_id = ? ORDER BY started_at",
        (trace_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["attributes"] = json.loads(d.pop("attributes_json") or "{}")
        out.append(d)
    return out
