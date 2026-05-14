from schemas import PageContent, ProcessedDocument, StructuredFields
from retrieval.chunker import chunk_document


def _doc(text: str) -> ProcessedDocument:
    return ProcessedDocument(
        doc_id="doc1",
        filename="x.pdf",
        pages=[PageContent(page=1, text=text, ocr_confidence=0.9, source="digital")],
        fields=StructuredFields(),
        full_text=text,
    )


def test_short_text_yields_one_chunk_with_correct_metadata():
    chunks = chunk_document(_doc("Hello world.\n\nSecond paragraph."))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.doc_id == "doc1"
    assert c.page == 1
    assert c.chunk_id.startswith("doc1:p1:c")
    assert c.ocr_confidence == 0.9


def test_long_text_splits_with_overlap_preserving_provenance():
    paragraph = "alpha " * 200  # ~200 words
    text = "\n\n".join([paragraph] * 4)
    chunks = chunk_document(_doc(text))
    assert len(chunks) >= 2
    # every chunk keeps page provenance
    assert all(c.page == 1 for c in chunks)
    # chunk ids are unique within doc
    assert len({c.chunk_id for c in chunks}) == len(chunks)
