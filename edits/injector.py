"""Build the `extra_system` slot the generator accepts.

Composes two blocks: (1) numbered style rules drawn from the rules table,
(2) one or two exemplar drafts the operator previously approved. The
generator concatenates this onto its system prompt verbatim.
"""
from __future__ import annotations
from edits.rules import top_rules
from edits.bank import get_exemplars


def build_feedback_system(doc_type: str | None) -> str:
    rules = top_rules(doc_type=doc_type)
    exemplars = get_exemplars(doc_type=doc_type)
    parts: list[str] = []

    if rules:
        lines = ["LEARNED STYLE RULES (from prior operator edits — follow these):"]
        for i, r in enumerate(rules, 1):
            lines.append(f"{i}. [{r['category']}] {r['rule']}  (support={r['support']})")
        parts.append("\n".join(lines))

    if exemplars:
        parts.append("EXEMPLAR DRAFTS (operator-approved, match this tone and structure):")
        for i, ex in enumerate(exemplars, 1):
            parts.append(f"--- exemplar {i} ---\n{ex}")

    return "\n\n".join(parts)
