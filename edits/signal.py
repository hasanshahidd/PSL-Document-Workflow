"""Turn raw edit operations into reusable signals (via the LLM provider)."""
from __future__ import annotations
import json
from typing import Iterable
from providers.registry import get_llm
from edits.diff import EditOp


CATEGORIES = [
    "phrasing_swap",
    "section_added",
    "section_removed",
    "boilerplate_stripped",
    "citation_corrected",
    "tone_shift",
    "fact_corrected",
    "other",
]


SYSTEM = (
    "You analyse a single sentence-level edit on a legal-style draft and "
    "extract a reusable rule for future drafts. "
    "Reply with JSON only: "
    '{"category": <one of ' + ", ".join(CATEGORIES) + '>, '
    '"rule": <short imperative <= 25 words, or null if not generalisable>, '
    '"reason": <short reason for the category>}'
)


def classify_edit(op: EditOp, doc_type: str | None = None) -> dict:
    if op.op == "kept":
        return {"category": "kept", "rule": None, "reason": "no change"}

    user = json.dumps(
        {
            "operation": op.op,
            "original": op.original,
            "edited": op.edited,
            "doc_type": doc_type,
        },
        ensure_ascii=False,
    )
    try:
        resp = get_llm().complete(
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=256,
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        if parsed.get("category") not in CATEGORIES:
            parsed["category"] = "other"
        return parsed
    except Exception as e:
        return {"category": "other", "rule": None, "reason": f"classify_failed: {e}"}


def classify_all(ops: Iterable[EditOp], doc_type: str | None = None) -> list[dict]:
    out: list[dict] = []
    for op in ops:
        if op.op == "kept":
            continue
        signal = classify_edit(op, doc_type=doc_type)
        signal["op"] = op.op
        signal["original"] = op.original
        signal["edited"] = op.edited
        out.append(signal)
    return out
