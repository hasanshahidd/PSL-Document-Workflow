"""Exemplar bank: store accepted edited drafts as few-shot examples.

We index by doc_type so a future draft for the same kind of document can be
shown a couple of operator-approved precedents. We cap the bank size by
doc_type and prefer the most recent exemplars.
"""
from __future__ import annotations
from storage.db import get_conn


MAX_EXEMPLARS_PER_DOCTYPE = 5
EXEMPLARS_INJECTED = 2


def add_exemplar(edited_text: str, doc_type: str | None, source_draft_id: str | None) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO exemplars(doc_type, source_draft_id, edited_text) VALUES(?,?,?)",
        (doc_type, source_draft_id, edited_text),
    )
    # prune older ones beyond the cap
    conn.execute(
        """
        DELETE FROM exemplars
        WHERE exemplar_id IN (
            SELECT exemplar_id FROM exemplars
            WHERE COALESCE(doc_type, '') = COALESCE(?, '')
            ORDER BY created_at DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (doc_type, MAX_EXEMPLARS_PER_DOCTYPE),
    )
    conn.commit()


def get_exemplars(doc_type: str | None, limit: int = EXEMPLARS_INJECTED) -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT edited_text FROM exemplars
        WHERE COALESCE(doc_type, '') = COALESCE(?, '')
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (doc_type, limit),
    ).fetchall()
    return [r["edited_text"] for r in rows]
