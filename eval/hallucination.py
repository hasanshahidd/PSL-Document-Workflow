"""Ablation: count unsupported sentences with vs without grounding constraints."""
from __future__ import annotations
import uuid
from providers.registry import get_llm
from schemas import Draft
from drafting.generator import gather_evidence
from drafting.prompt import GENERATOR_SYSTEM, build_generator_user_message
from drafting.citer import parse_draft


BASELINE_SYSTEM = (
    "You write a Case Fact Summary from the provided source material. "
    "Use the same section headings: Parties, Timeline, Subject Matter, "
    "Material Facts, Uncertain or Unclear."
)


def _generate(system: str, evidence, doc_ids: list[str]) -> Draft:
    resp = get_llm().complete(
        system=system,
        messages=[{"role": "user", "content": build_generator_user_message(evidence)}],
        max_tokens=2048,
    )
    sentences, uncertain = parse_draft(resp.text, evidence)
    return Draft(
        draft_id=str(uuid.uuid4()),
        draft_type="case_fact_summary",
        doc_ids=doc_ids,
        sentences=sentences,
        uncertain_sections=uncertain,
    )


def _unsupported_count(d: Draft) -> int:
    return sum(1 for s in d.sentences if not s.supported and not s.text.startswith("#"))


def run_ablation(doc_ids: list[str]) -> dict:
    evidence = gather_evidence(doc_ids)
    grounded = _generate(GENERATOR_SYSTEM, evidence, doc_ids)
    baseline = _generate(BASELINE_SYSTEM, evidence, doc_ids)
    return {
        "grounded": {
            "n_sentences": len(grounded.sentences),
            "n_unsupported": _unsupported_count(grounded),
            "grounding_rate": round(grounded.grounding_rate, 4),
        },
        "baseline": {
            "n_sentences": len(baseline.sentences),
            "n_unsupported": _unsupported_count(baseline),
            "grounding_rate": round(baseline.grounding_rate, 4),
        },
    }
