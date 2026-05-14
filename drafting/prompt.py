"""Shared prompt fragments used by the drafting pipeline.

Section lists and writing instructions per template live in
`drafting/templates.py`. This file only keeps the cross template bits.

GENERATOR_SYSTEM and build_generator_user_message are retained for use by
the hallucination ablation eval, which compares one shot grounded
generation against a baseline. The live drafting flow does not use them
any more.
"""
from __future__ import annotations
from schemas import EvidenceChunk


PLANNER_SYSTEM = (
    "You generate retrieval queries for a document summariser. "
    "Given a section heading and the list of source documents, output 1-2 "
    "short search queries (each <15 words) likely to retrieve relevant "
    "passages. Reply with JSON: {\"queries\": [<str>, ...]}."
)


GENERATOR_SYSTEM = """You write a Case Fact Summary grounded strictly in the provided evidence.

RULES. Non negotiable.

1. Every factual sentence MUST end with one or more citations in the form
   [chunk_id]. You may ONLY cite chunk_ids that appear in the ALLOWED
   CITATIONS list at the top of the user message. Any chunk_id not in
   that list is forbidden, even if it looks plausible. Do not invent,
   guess, abbreviate, or extrapolate chunk_ids.
2. If a section has no supporting evidence in the ALLOWED CITATIONS list,
   write exactly: "No supporting evidence found in the provided documents."
   for that section.
3. Do NOT invent names, dates, amounts, or quotes. If the evidence is
   ambiguous, place the relevant point under "Uncertain or Unclear"
   instead.
4. Quote sparingly. Paraphrase the evidence in your own words but stay
   faithful to it.
5. Use the exact section headings provided. Do not add new sections.

Output format (markdown):

## Parties
<sentences with citations>

## Timeline
<sentences with citations>

## Subject Matter
<sentences with citations>

## Material Facts
<sentences with citations>

## Uncertain or Unclear
<sentences with citations, or "No supporting evidence found...">
"""


def format_evidence(chunks: list[EvidenceChunk]) -> str:
    lines = []
    for c in chunks:
        conf_tag = "" if c.ocr_confidence >= 0.85 else f" (OCR conf {c.ocr_confidence:.2f})"
        lines.append(f"[{c.chunk_id}] (doc={c.doc_id}, page={c.page}){conf_tag}\n{c.text}")
    return "\n\n---\n\n".join(lines)


def build_generator_user_message(chunks: list[EvidenceChunk]) -> str:
    allowed = "\n".join(f"  - {c.chunk_id}" for c in chunks) or "  (none)"
    return (
        "ALLOWED CITATIONS. You may cite ONLY these chunk_ids. Any other id "
        "is forbidden.\n"
        f"{allowed}\n\n"
        "=====\n\n"
        "Evidence chunks follow. Each is labelled with its citation id.\n\n"
        f"{format_evidence(chunks)}\n\n"
        "Write the Case Fact Summary now, following every rule. "
        "Every citation must come from the ALLOWED CITATIONS list above."
    )
