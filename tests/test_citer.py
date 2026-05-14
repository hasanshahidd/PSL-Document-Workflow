from schemas import EvidenceChunk
from drafting.citer import parse_draft


def _ev(cid: str) -> EvidenceChunk:
    return EvidenceChunk(chunk_id=cid, doc_id="d1", page=1, text="x", score=0.9)


def test_sentences_without_citations_are_marked_unsupported():
    text = (
        "## Parties\n"
        "Acme is the plaintiff [d1:p1:c0].\n"
        "Beta Corp signed the contract on Tuesday.\n"
    )
    sentences, _ = parse_draft(text, [_ev("d1:p1:c0")])
    body = [s for s in sentences if not s.text.startswith("#")]
    assert body[0].supported is True
    assert body[0].citations[0].chunk_id == "d1:p1:c0"
    assert body[1].supported is False
    assert body[1].citations == []


def test_no_evidence_sentinel_counts_as_supported():
    from drafting.citer import SENTINEL_NO_EVIDENCE
    sentences, _ = parse_draft(f"## Uncertain or Unclear\n{SENTINEL_NO_EVIDENCE}", [])
    body = [s for s in sentences if not s.text.startswith("#")]
    assert body and body[0].supported is True


def test_uncertain_section_collects_uncited_sentences():
    text = (
        "## Uncertain or Unclear\n"
        "It is unclear whether the option was exercised."
    )
    _, uncertain = parse_draft(text, [])
    assert any("unclear" in u.lower() for u in uncertain)
