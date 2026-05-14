"""Self review pass for grounded drafts.

After the generator produces a draft, we ask the LLM (in a strict judge
role) to read each cited sentence and decide whether the cited chunk
actually supports the claim. Sentences that fail are moved to the
'Uncertain or Unclear' section instead of being shown as confident
output. This is the same idea as the eval grounding metric, but executed
inside the generation loop so the operator never sees unsupported claims
in regular sections.

The reviewer also extracts a short supporting quote per cited sentence
(the actual phrase from the chunk that backs the claim). The quote is
attached to the citation so the UI can show it on hover.
"""
from __future__ import annotations
import json
from providers.registry import get_llm, get_vector_store
from schemas import Citation, Draft, DraftSentence


REVIEWER_SYSTEM = (
    "You are a strict grounding judge. For a claim and its cited evidence, "
    "decide whether the evidence actually supports the claim and, if so, "
    "quote the shortest supporting phrase from the evidence verbatim. "
    "Reply JSON only: "
    '{"supported": true|false, "quote": <verbatim phrase or null>, "reason": <short>}.'
)


def _judge(claim: str, evidence: str) -> dict:
    user = json.dumps({"claim": claim, "evidence": evidence}, ensure_ascii=False)
    try:
        resp = get_llm().complete(
            system=REVIEWER_SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=192,
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"supported": False, "quote": None, "reason": f"judge_failed: {e}"}


def review_and_annotate(draft: Draft) -> Draft:
    """Re check every cited sentence and attach the supporting quote.

    Sentences that the judge marks unsupported keep their text but flip
    `supported=False`. The draft's `uncertain_sections` is appended with
    those texts so the UI can also surface them under Uncertain.
    """
    store = get_vector_store()
    new_sentences: list[DraftSentence] = []
    newly_unsupported: list[str] = []

    for s in draft.sentences:
        # headings, sentinels, and already unsupported sentences are passed through
        if s.text.startswith("#") or not s.citations:
            new_sentences.append(s)
            continue

        new_citations: list[Citation] = []
        any_supported = False
        for cit in s.citations:
            chunk_text = store.get_text(cit.chunk_id) or ""
            verdict = _judge(s.text, chunk_text)
            quote = verdict.get("quote")
            supported = bool(verdict.get("supported"))
            new_citations.append(
                Citation(
                    doc_id=cit.doc_id, page=cit.page, chunk_id=cit.chunk_id,
                    quote=quote if isinstance(quote, str) and quote.strip() else None,
                )
            )
            any_supported = any_supported or supported

        new_sentences.append(
            DraftSentence(
                idx=s.idx, text=s.text, citations=new_citations,
                supported=any_supported,
            )
        )
        if not any_supported:
            newly_unsupported.append(s.text)

    return Draft(
        draft_id=draft.draft_id,
        draft_type=draft.draft_type,
        doc_ids=draft.doc_ids,
        sentences=new_sentences,
        uncertain_sections=draft.uncertain_sections + newly_unsupported,
    )
