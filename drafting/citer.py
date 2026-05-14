"""Post-hoc citation parser and validator.

Splits a generated draft into sentences, extracts any [chunk_id] citations,
and marks sentences without citations as unsupported. Headings and the
'No supporting evidence...' sentinel are exempt from the rule.
"""
from __future__ import annotations
import re
from schemas import Citation, DraftSentence, EvidenceChunk


CITATION_RE = re.compile(r"\[([A-Za-z0-9_:.\-]+)\]")
SENTINEL_NO_EVIDENCE = "No supporting evidence found in the provided documents."


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter that keeps headings as their own units."""
    out: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            out.append(block)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", block)
        out.extend(p.strip() for p in parts if p.strip())
    return out


def parse_draft(
    text: str,
    evidence: list[EvidenceChunk],
) -> tuple[list[DraftSentence], list[str]]:
    """Return (sentences, uncertain_section_lines)."""
    chunk_lookup = {c.chunk_id: c for c in evidence}
    sentences: list[DraftSentence] = []
    uncertain: list[str] = []
    current_section: str | None = None

    for i, sent in enumerate(_split_sentences(text)):
        is_heading = sent.startswith("#")
        if is_heading:
            current_section = sent.lstrip("#").strip().lower()
            sentences.append(DraftSentence(idx=i, text=sent, citations=[], supported=True))
            continue

        citation_ids = CITATION_RE.findall(sent)
        citations = [
            Citation(doc_id=chunk_lookup[cid].doc_id, page=chunk_lookup[cid].page, chunk_id=cid)
            for cid in citation_ids
            if cid in chunk_lookup
        ]

        supported = bool(citations) or sent.strip() == SENTINEL_NO_EVIDENCE
        sentences.append(
            DraftSentence(idx=i, text=sent, citations=citations, supported=supported)
        )
        if current_section and "uncertain" in current_section and sent.strip() != SENTINEL_NO_EVIDENCE:
            uncertain.append(sent)

    return sentences, uncertain
