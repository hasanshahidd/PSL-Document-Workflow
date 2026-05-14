"""Structured JSON logging with trace_id propagation.

Every log record carries the active trace_id (set by obs.tracer when a span
is open) and is emitted as a single line of JSON. Reviewers can pipe the
output through jq, and the same lines can be ingested by a log aggregator
without any reshape.
"""
from __future__ import annotations
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)


def set_trace_id(value: str | None) -> None:
    _trace_id.set(value)


def set_span_id(value: str | None) -> None:
    _span_id.set(value)


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_span_id() -> str | None:
    return _span_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        tid = _trace_id.get()
        sid = _span_id.get()
        if tid:
            payload["trace_id"] = tid
        if sid:
            payload["span_id"] = sid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                payload[k[4:]] = v
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def setup(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup()
    return logging.getLogger(name)
