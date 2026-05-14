"""Cross encoder reranker.

A bi encoder embedding model is fast but coarse. We retrieve the top K
candidates by cosine similarity, then re score each (query, chunk) pair
with a cross encoder that reads both texts together. This is the single
largest precision win in modern RAG and costs only one extra model pass
over the K candidates.

Model. `cross-encoder/ms-marco-MiniLM-L-6-v2` is a 22 MB checkpoint trained
on MS MARCO passage ranking. Open weights, no API key, runs on CPU.
"""
from __future__ import annotations
from sentence_transformers import CrossEncoder
from config import settings
from obs.tracer import span
from schemas import EvidenceChunk


_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        name = getattr(settings, "reranker_model", _DEFAULT_MODEL) or _DEFAULT_MODEL
        _model = CrossEncoder(name)
    return _model


def rerank(query: str, chunks: list[EvidenceChunk], top_n: int | None = None) -> list[EvidenceChunk]:
    """Re score chunks against the query with a cross encoder.

    Returns chunks sorted by the cross encoder score (descending), with the
    new score stored back on `chunk.score`. The original embedding score is
    blended 80/20 (cross encoder dominates) so OCR confidence weighting from
    the previous stage is not entirely lost.
    """
    if not chunks:
        return []
    with span("rerank", query=query[:80], n_in=len(chunks)) as s:
        pairs = [(query, c.text) for c in chunks]
        scores = _get_model().predict(pairs, show_progress_bar=False)
        rescored: list[EvidenceChunk] = []
        for chunk, ce_score in zip(chunks, scores):
            # Cross encoder scores can be negative. Squash to roughly [0,1]
            # via a sigmoid so blending with cosine similarity is meaningful.
            import math
            ce_norm = 1.0 / (1.0 + math.exp(-float(ce_score)))
            blended = 0.8 * ce_norm + 0.2 * chunk.score
            rescored.append(
                EvidenceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page=chunk.page,
                    text=chunk.text,
                    ocr_confidence=chunk.ocr_confidence,
                    score=round(blended, 4),
                )
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        if top_n is not None:
            rescored = rescored[:top_n]
        s.set_many(n_out=len(rescored), top_score=rescored[0].score if rescored else 0.0)
        return rescored
