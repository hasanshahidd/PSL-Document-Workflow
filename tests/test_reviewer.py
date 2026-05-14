"""Reviewer self review pass.

Uses the fake LLM provider to script grounding judge verdicts and
verifies that:
  - sentences whose citation is judged unsupported flip to supported=False
  - the supporting quote returned by the judge is attached to the citation
  - headings and 'no supporting evidence' sentinels are left alone
"""
from __future__ import annotations
from schemas import Citation, Draft, DraftSentence
from providers.base import EmbeddingProvider, VectorStore, StoredChunk
from providers.registry import set_embedder, set_vector_store


class _Embedder(EmbeddingProvider):
    @property
    def name(self): return "x"
    def embed(self, texts): return [[0.0] for _ in texts]


class _Store(VectorStore):
    def __init__(self, by_id):
        self._rows = by_id
    def upsert(self, items): return 0
    def query(self, embedding, k=5, doc_ids=None): return []
    def get_text(self, chunk_id): return self._rows.get(chunk_id)
    def delete_doc(self, doc_id): return 0
    def reset(self): self._rows.clear()


def test_unsupported_citation_flips_supported_to_false(fake_llm):
    set_embedder(_Embedder())
    set_vector_store(_Store({"d1:p1:c0": "Acme entered into agreement on Jan 1 2026."}))

    fake_llm.queue('{"supported": false, "quote": null, "reason": "not actually about parties"}')

    draft = Draft(
        draft_id="x",
        draft_type="case_fact_summary",
        doc_ids=["d1"],
        sentences=[
            DraftSentence(idx=0, text="## Parties", citations=[], supported=True),
            DraftSentence(idx=1, text="Acme is the plaintiff.",
                          citations=[Citation(doc_id="d1", page=1, chunk_id="d1:p1:c0")],
                          supported=True),
        ],
    )

    from drafting.reviewer import review_and_annotate
    reviewed = review_and_annotate(draft)

    body = [s for s in reviewed.sentences if not s.text.startswith("#")]
    assert body[0].supported is False
    assert "Acme is the plaintiff." in reviewed.uncertain_sections


def test_quote_is_attached_when_judge_supplies_one(fake_llm):
    set_embedder(_Embedder())
    set_vector_store(_Store({"d1:p1:c0": "The agreement took effect on 15 January 2026."}))

    fake_llm.queue(
        '{"supported": true, "quote": "took effect on 15 January 2026", "reason": "exact date match"}'
    )

    draft = Draft(
        draft_id="x",
        draft_type="case_fact_summary",
        doc_ids=["d1"],
        sentences=[
            DraftSentence(idx=0, text="The agreement is dated January 2026.",
                          citations=[Citation(doc_id="d1", page=1, chunk_id="d1:p1:c0")],
                          supported=True),
        ],
    )

    from drafting.reviewer import review_and_annotate
    reviewed = review_and_annotate(draft)

    sent = reviewed.sentences[0]
    assert sent.supported is True
    assert sent.citations[0].quote == "took effect on 15 January 2026"


def test_headings_are_passed_through_unchanged(fake_llm):
    # no fake response queued; reviewer should not call the LLM for headings
    set_embedder(_Embedder())
    set_vector_store(_Store({}))

    draft = Draft(
        draft_id="x",
        draft_type="case_fact_summary",
        doc_ids=["d1"],
        sentences=[
            DraftSentence(idx=0, text="## Parties", citations=[], supported=True),
        ],
    )

    from drafting.reviewer import review_and_annotate
    reviewed = review_and_annotate(draft)

    assert reviewed.sentences[0].supported is True
    assert reviewed.sentences[0].text == "## Parties"
