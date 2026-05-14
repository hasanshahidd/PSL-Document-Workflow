"""Index shim — wraps the VectorStore provider with the legacy dict-shaped API.

Existing callers keep working. New code can call `get_vector_store()` directly
for the typed API.
"""
from __future__ import annotations
from providers.registry import get_embedder, get_vector_store
from schemas import Chunk


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    vectors = get_embedder().embed([c.text for c in chunks])
    items = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "page": c.page,
            "text": c.text,
            "ocr_confidence": c.ocr_confidence,
            "embedding": v,
        }
        for c, v in zip(chunks, vectors)
    ]
    return get_vector_store().upsert(items)


def query(text: str, k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
    vec = get_embedder().embed([text])[0]
    hits = get_vector_store().query(vec, k=k, doc_ids=doc_ids)
    return [
        {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "page": h.page,
            "text": h.text,
            "ocr_confidence": h.ocr_confidence,
            "score": h.score,
        }
        for h in hits
    ]


def reset() -> None:
    get_vector_store().reset()


# Legacy export the eval/grounding module reaches for.
def _get_collection():
    """Compat shim for callers that want raw collection access."""
    return get_vector_store()._collection  # type: ignore[attr-defined]
