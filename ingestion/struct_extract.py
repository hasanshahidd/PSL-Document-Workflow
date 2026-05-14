"""Extract structured fields (doc_type, parties, dates, matter_id) from raw text.

A cheap regex pass catches obvious dates and matter IDs. Everything that needs
judgement goes to the LLM provider in JSON mode.
"""
from __future__ import annotations
import json
import re
from config import settings
from providers.registry import get_llm
from schemas import StructuredFields


DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
MATTER_RE = re.compile(r"\b(?:Matter|Case|File)\s*(?:No\.?|#)?\s*([A-Z0-9\-]{4,})", re.IGNORECASE)


SYSTEM = (
    "Extract structured fields from a document. Reply with JSON only: "
    '{"doc_type": <short snake_case label or null>, "parties": [<names>], '
    '"key_dates_iso": [<YYYY-MM-DD>]}. '
    "doc_type examples (pick the closest one). "
    "Legal: contract, notice, complaint, affidavit, title_report, memo, "
    "subpoena, stipulation, deposition, indictment, injunction, pleading, "
    "amendment, addendum, agreement. "
    "Technical: benchmark, standard, manual, guide, report, specification, "
    "policy, whitepaper, rfc, handbook, playbook, runbook, framework. "
    "Reply null only when none of these fit."
)


def extract_fields(text: str) -> StructuredFields:
    fields = StructuredFields()
    fields.dates = list(dict.fromkeys(DATE_RE.findall(text)))[:20]
    m = MATTER_RE.search(text)
    if m:
        fields.matter_id = m.group(1)

    # Skip the LLM call entirely when no provider is configured.
    has_key = bool(settings.openai_api_key or settings.anthropic_api_key)
    if not has_key:
        return fields

    snippet = text[:6000]
    try:
        resp = get_llm().complete(
            system=SYSTEM,
            messages=[{"role": "user", "content": snippet}],
            max_tokens=512,
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        fields.doc_type = parsed.get("doc_type")
        fields.parties = parsed.get("parties", [])[:10]
        extra_dates = parsed.get("key_dates_iso", [])
        if extra_dates:
            fields.dates = list(dict.fromkeys(fields.dates + extra_dates))[:20]
    except Exception:
        pass
    return fields
