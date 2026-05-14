"""Aggregate edit signals into style rules persisted in SQLite.

A signal's `rule` string is the unique key. When the same rule is seen
again we just bump its support count and last_seen timestamp. Rules with
support >= MIN_SUPPORT become eligible for injection into the next draft.
"""
from __future__ import annotations
from storage.db import get_conn


MIN_SUPPORT_FOR_INJECTION = 2
MAX_RULES_INJECTED = 8


def ingest_signals(signals: list[dict], doc_type: str | None = None) -> int:
    """Persist each generalisable signal as (or update) a style rule."""
    conn = get_conn()
    n = 0
    for s in signals:
        rule = s.get("rule")
        category = s.get("category", "other")
        if not rule:
            continue
        existing = conn.execute(
            "SELECT rule_id, support FROM style_rules WHERE rule = ?", (rule,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE style_rules SET support = support + 1, last_seen = datetime('now') "
                "WHERE rule_id = ?",
                (existing["rule_id"],),
            )
        else:
            conn.execute(
                "INSERT INTO style_rules(category, rule, doc_type, support) VALUES(?,?,?,1)",
                (category, rule, doc_type),
            )
        n += 1
    conn.commit()
    return n


def top_rules(doc_type: str | None = None, limit: int = MAX_RULES_INJECTED) -> list[dict]:
    """Return rules eligible for injection, doc-type-matched first."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT category, rule, doc_type, support
        FROM style_rules
        WHERE support >= ?
        ORDER BY (doc_type = ?) DESC, support DESC, last_seen DESC
        LIMIT ?
        """,
        (MIN_SUPPORT_FOR_INJECTION, doc_type, limit),
    ).fetchall()
    return [dict(r) for r in rows]
