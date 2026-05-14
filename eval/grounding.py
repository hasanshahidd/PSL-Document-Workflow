"""LLM-as-judge grounding score for a generated draft."""
from __future__ import annotations
import json
from obs.tracer import span
from providers.registry import get_vector_store, get_llm
from schemas import Draft


JUDGE_SYSTEM = (
    "You are a strict grounding judge. Given a CLAIM and EVIDENCE, decide if "
    "the evidence supports the claim. "
    'Reply JSON: {"supported": true|false, "reason": <short>}.'
)


def judge_sentence(claim: str, chunk_id: str) -> dict:
    evidence = get_vector_store().get_text(chunk_id) or ""
    user = json.dumps({"claim": claim, "evidence": evidence}, ensure_ascii=False)
    try:
        resp = get_llm().complete(
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=128,
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"supported": False, "reason": f"judge_failed: {e}"}


def score_draft(draft: Draft) -> dict:
    cited = [s for s in draft.sentences if s.citations and not s.text.startswith("#")]
    if not cited:
        return {"n_cited": 0, "n_supported": 0, "grounding_score": 0.0, "verdicts": []}

    with span("eval.grounding", n_cited=len(cited)) as s:
        verdicts = []
        n_supported = 0
        for sent in cited:
            cid = sent.citations[0].chunk_id
            v = judge_sentence(sent.text, cid)
            verdicts.append({"sentence": sent.text, "chunk_id": cid, **v})
            if v.get("supported"):
                n_supported += 1
        score = n_supported / len(cited)
        s.set("grounding_score", score)
        return {
            "n_cited": len(cited),
            "n_supported": n_supported,
            "grounding_score": score,
            "verdicts": verdicts,
        }
