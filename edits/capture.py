"""Orchestrate edit capture: diff -> classify -> persist signals, rules, exemplar."""
from __future__ import annotations
import json
from config import settings
from schemas import Draft, ProcessedDocument
from storage.db import get_conn
from storage.drafts_store import get_draft
from edits.diff import align, summary
from edits.signal import classify_all
from edits.rules import ingest_signals
from edits.bank import add_exemplar


def _doc_type_for_draft(draft: Draft) -> str | None:
    for doc_id in draft.doc_ids:
        path = settings.processed_dir / f"{doc_id}.json"
        if not path.exists():
            continue
        doc = ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if doc.fields.doc_type:
            return doc.fields.doc_type
    return None


def capture_edit(draft_id: str, edited_text: str) -> dict:
    draft = get_draft(draft_id)
    if draft is None:
        raise ValueError(f"unknown draft_id: {draft_id}")

    doc_type = _doc_type_for_draft(draft)
    ops = align(draft.text, edited_text)
    signals = classify_all(ops, doc_type=doc_type)
    n_rules = ingest_signals(signals, doc_type=doc_type)
    add_exemplar(edited_text=edited_text, doc_type=doc_type, source_draft_id=draft_id)

    conn = get_conn()
    conn.execute(
        "INSERT INTO edits(draft_id, edited_text, diff_json, signals_json) VALUES(?,?,?,?)",
        (
            draft_id,
            edited_text,
            json.dumps([op.__dict__ for op in ops]),
            json.dumps(signals),
        ),
    )
    conn.commit()
    return {
        "draft_id": draft_id,
        "doc_type": doc_type,
        "diff_summary": summary(ops),
        "signals_recorded": len(signals),
        "rules_updated": n_rules,
    }
