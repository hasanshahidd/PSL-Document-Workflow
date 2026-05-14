"""SQLite persistence for drafts, edits, learned style rules, and exemplars.

A single connection with WAL mode is plenty for this workflow. Schema is
created on first call to get_conn().
"""
from __future__ import annotations
import sqlite3
import threading
import config  # use the module so tests can rebind config.settings


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    draft_id TEXT PRIMARY KEY,
    draft_type TEXT NOT NULL,
    doc_ids TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edits (
    edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id TEXT NOT NULL,
    edited_text TEXT NOT NULL,
    diff_json TEXT,
    signals_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (draft_id) REFERENCES drafts(draft_id)
);

CREATE TABLE IF NOT EXISTS style_rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    rule TEXT NOT NULL UNIQUE,
    doc_type TEXT,
    support INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exemplars (
    exemplar_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_type TEXT,
    source_draft_id TEXT,
    edited_text TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traces (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    status TEXT NOT NULL,
    attributes_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces(trace_id);

CREATE TABLE IF NOT EXISTS cost_ledger (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT,
    span_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    usd REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cost_trace_id ON cost_ledger(trace_id);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            path = config.settings.data_dir / "app.db"
            _conn = sqlite3.connect(str(path), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
            _conn.commit()
    return _conn
