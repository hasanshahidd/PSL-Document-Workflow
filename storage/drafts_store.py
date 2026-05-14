"""Persist and fetch Draft objects."""
from __future__ import annotations
import json
from schemas import Draft
from storage.db import get_conn


def save_draft(draft: Draft) -> None:
    get_conn().execute(
        "INSERT OR REPLACE INTO drafts(draft_id, draft_type, doc_ids, body) VALUES(?,?,?,?)",
        (draft.draft_id, draft.draft_type, json.dumps(draft.doc_ids), draft.model_dump_json()),
    )
    get_conn().commit()


def get_draft(draft_id: str) -> Draft | None:
    row = get_conn().execute(
        "SELECT body FROM drafts WHERE draft_id=?", (draft_id,)
    ).fetchone()
    return Draft.model_validate_json(row["body"]) if row else None
