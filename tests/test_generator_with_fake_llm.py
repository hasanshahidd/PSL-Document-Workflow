"""End to end test of the per section pipeline with fake providers.

Proves the architecture is provider pluggable. Zero network calls.

The new generator pipeline makes these LLM calls per section:
  1. planner queries
  2. section body (with allowed citations)
And then a reviewer judge call per cited sentence at the end.
"""
from __future__ import annotations
from providers.base import EmbeddingProvider, VectorStore, StoredChunk
from providers.registry import set_embedder, set_vector_store


class HashEmbedder(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "hash-embedder"

    def embed(self, texts):
        return [[float(len(t) % 7), float(sum(ord(c) for c in t) % 13)] for t in texts]


class MemStore(VectorStore):
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def upsert(self, items):
        for it in items:
            self._rows[it["chunk_id"]] = it
        return len(items)

    def query(self, embedding, k=5, doc_ids=None):
        rows = list(self._rows.values())
        if doc_ids:
            rows = [r for r in rows if r["doc_id"] in doc_ids]
        return [
            StoredChunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], page=r["page"],
                text=r["text"], score=0.9, ocr_confidence=r.get("ocr_confidence", 1.0),
            )
            for r in rows[:k]
        ]

    def get_text(self, chunk_id):
        r = self._rows.get(chunk_id)
        return r["text"] if r else None

    def delete_doc(self, doc_id):
        keys = [k for k, v in self._rows.items() if v.get("doc_id") == doc_id]
        for k in keys:
            del self._rows[k]
        return len(keys)

    def reset(self):
        self._rows.clear()


def test_full_draft_flow_uses_fakes_and_produces_cited_draft(fake_llm, monkeypatch):
    set_embedder(HashEmbedder())
    set_vector_store(MemStore())

    # Skip the real cross encoder. Replace `rerank` with a passthrough so the
    # test stays hermetic and fast.
    import retrieval.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, chunks, top_n=None: chunks[: top_n or len(chunks)])
    # The per_section module imported the function at module import time,
    # so monkey patch that binding too.
    import drafting.per_section as ps
    monkeypatch.setattr(ps, "rerank", lambda q, chunks, top_n=None: chunks[: top_n or len(chunks)])

    from schemas import PageContent, ProcessedDocument, StructuredFields
    from retrieval.chunker import chunk_document
    from retrieval import index

    doc = ProcessedDocument(
        doc_id="docA",
        filename="a.pdf",
        pages=[PageContent(page=1, text="Acme entered into an agreement with Beta on Jan 1 2026.",
                           ocr_confidence=0.95, source="digital")],
        fields=StructuredFields(doc_type="contract"),
        full_text="Acme entered into an agreement with Beta on Jan 1 2026.",
    )
    chunks = chunk_document(doc)
    assert index.upsert_chunks(chunks) == len(chunks)
    chunk_id = chunks[0].chunk_id

    fake_llm.queue(
        # === Section 1: Parties ===
        '{"queries":["who are the parties"]}',                       # planner
        f"Acme and Beta are parties [{chunk_id}].",                  # generator
        # === Section 2: Timeline ===
        '{"queries":["key dates"]}',
        f"Agreement signed Jan 1 2026 [{chunk_id}].",
        # === Section 3: Subject Matter ===
        '{"queries":["subject matter"]}',
        f"Contract between Acme and Beta [{chunk_id}].",
        # === Section 4: Material Facts ===
        '{"queries":["material facts"]}',
        f"The agreement exists [{chunk_id}].",
        # === Section 5: Uncertain ===
        '{"queries":["uncertain"]}',
        "No supporting evidence found in the provided documents.",
        # === Reviewer judges (4 cited sentences, supported true) ===
        '{"supported": true, "quote": "Acme entered into an agreement with Beta", "reason": "match"}',
        '{"supported": true, "quote": "Jan 1 2026", "reason": "match"}',
        '{"supported": true, "quote": "agreement with Beta", "reason": "match"}',
        '{"supported": true, "quote": "agreement", "reason": "match"}',
    )

    from drafting.generator import generate_case_fact_summary
    draft = generate_case_fact_summary(["docA"], use_feedback=False, persist=False)

    body = [s for s in draft.sentences if not s.text.startswith("#")]
    # 4 real body sentences + 1 sentinel = 5
    assert len(body) >= 4
    cited = [s for s in body if s.citations]
    assert len(cited) == 4
    assert all(s.supported for s in cited)
    assert all(s.citations[0].quote for s in cited)
