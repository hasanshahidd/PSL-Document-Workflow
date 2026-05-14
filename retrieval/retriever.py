"""Confidence-weighted retrieval.

Raw cosine similarity is multiplied by an OCR-confidence factor so a chunk
that lives on a partially garbled page is preferred only when nothing else
clears the bar. The weighting is:

    weighted_score = similarity * (CONF_FLOOR + (1 - CONF_FLOOR) * ocr_confidence)

CONF_FLOOR keeps low-confidence chunks competitive when they're the only
option (e.g. handwritten-only documents).
"""
from __future__ import annotations
from obs.tracer import span
from schemas import EvidenceChunk
from retrieval import index


CONF_FLOOR = 0.5


def _weighted(score: float, ocr_conf: float) -> float:
    return score * (CONF_FLOOR + (1.0 - CONF_FLOOR) * ocr_conf)


def retrieve(query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[EvidenceChunk]:
    with span("retrieve", query=query[:80], k=k) as s:
        hits = index.query(text=query, k=k, doc_ids=doc_ids)
        weighted = sorted(
            (
                EvidenceChunk(**{**h, "score": _weighted(h["score"], h["ocr_confidence"])})
                for h in hits
            ),
            key=lambda c: c.score,
            reverse=True,
        )
        s.set("returned", len(weighted))
        return weighted


def retrieve_for_drafting(
    queries: list[str],
    k_per_query: int = 4,
    doc_ids: list[str] | None = None,
) -> list[EvidenceChunk]:
    """Run multiple sub-queries (one per section) and de-duplicate by chunk_id."""
    with span("retrieve_for_drafting", n_queries=len(queries), k=k_per_query) as s:
        best: dict[str, EvidenceChunk] = {}
        for q in queries:
            for hit in retrieve(q, k=k_per_query, doc_ids=doc_ids):
                prev = best.get(hit.chunk_id)
                if prev is None or hit.score > prev.score:
                    best[hit.chunk_id] = hit
        ordered = sorted(best.values(), key=lambda c: c.score, reverse=True)
        s.set("evidence_chunks", len(ordered))
        return ordered
