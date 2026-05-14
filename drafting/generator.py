"""Grounded draft generation orchestrator.

Pipeline.

  1. Resolve the `doc_type` for the requested documents and pick the
     appropriate `DraftTemplate` (legal Case Fact Summary or Technical
     Document Summary, see drafting/templates.py).
  2. For each section in the template:
        plan retrieval queries
        bi encoder retrieve top K
        cross encoder rerank to top N
        generate that section ONLY with those N chunks as allowed citations
  3. Concatenate the per section bodies under their headings.
  4. Post hoc citer parses citations and flags any unsupported sentence.
  5. Self review pass: LLM as judge verifies each cited sentence against
     its chunk text. Sentences that fail flip to supported=False and are
     added to uncertain_sections. Each surviving citation receives a short
     supporting quote.
  6. Persist the draft. Its draft_id is also the trace_id so the full
     request tree is fetchable from /traces/{draft_id}.
"""
from __future__ import annotations
import uuid
from config import settings
from logging_setup import get_logger
from obs.tracer import span, start_trace
from schemas import Draft, EvidenceChunk, ProcessedDocument
from drafting.citer import parse_draft
from drafting.per_section import (
    generate_section, render_section, section_label_to_heading,
)
from drafting.reviewer import review_and_annotate
from drafting.templates import CASE_FACT_SUMMARY, DraftTemplate, select_template
from edits.injector import build_feedback_system
from storage.drafts_store import save_draft


log = get_logger(__name__)


def _doc_type_for(doc_ids: list[str]) -> str | None:
    for did in doc_ids:
        path = settings.processed_dir / f"{did}.json"
        if not path.exists():
            continue
        d = ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if d.fields.doc_type:
            return d.fields.doc_type
    return None


def gather_evidence(doc_ids: list[str], k_per_query: int = 4) -> list[EvidenceChunk]:
    """Legacy helper kept for the hallucination ablation eval. The new
    per section path retrieves on demand inside `generate_section`.
    Uses the Case Fact Summary template sections as a stable proxy.
    """
    from drafting.per_section import _retrieve_for_section
    pool: dict[str, EvidenceChunk] = {}
    for section in CASE_FACT_SUMMARY.sections:
        for c in _retrieve_for_section(section, doc_ids, k_initial=12, k_final=k_per_query):
            prev = pool.get(c.chunk_id)
            if prev is None or c.score > prev.score:
                pool[c.chunk_id] = c
    return sorted(pool.values(), key=lambda c: c.score, reverse=True)


def generate_case_fact_summary(
    doc_ids: list[str],
    extra_system: str | None = None,
    use_feedback: bool = True,
    persist: bool = True,
    run_self_review: bool = True,
    template: DraftTemplate | None = None,
) -> Draft:
    """Build a grounded summary using the per section pipeline.

    The function name is kept for backwards compatibility but the actual
    template is now chosen by `doc_type`. Caller can override with the
    `template` kwarg if they want a specific template regardless of type.
    """
    draft_id = str(uuid.uuid4())
    trace_id = start_trace(draft_id.replace("-", ""))

    doc_type = _doc_type_for(doc_ids)
    chosen_template = template or select_template(doc_type)

    with span("generate_case_fact_summary",
              draft_id=draft_id, doc_ids=doc_ids,
              template=chosen_template.name, doc_type=doc_type) as root:

        feedback_addendum = ""
        if extra_system is None and use_feedback:
            feedback_addendum = build_feedback_system(doc_type)
        elif extra_system:
            feedback_addendum = extra_system
        root.set("feedback_used", bool(feedback_addendum))

        # 1. Per section generation.
        section_bodies: list[str] = []
        combined_evidence: dict[str, EvidenceChunk] = {}
        for section in chosen_template.sections:
            body, evidence = generate_section(
                section, doc_ids, chosen_template,
                extra_system=feedback_addendum,
            )
            heading = section_label_to_heading(chosen_template, section)
            section_bodies.append(render_section(heading, body))
            for c in evidence:
                combined_evidence[c.chunk_id] = c
        root.set_many(
            sections=len(section_bodies),
            unique_chunks=len(combined_evidence),
        )

        # 2. Stitch and citer parse.
        full_text = "\n\n".join(section_bodies)
        sentences, uncertain = parse_draft(full_text, list(combined_evidence.values()))

        draft = Draft(
            draft_id=draft_id,
            draft_type=chosen_template.name,
            doc_ids=doc_ids,
            sentences=sentences,
            uncertain_sections=uncertain,
        )

        # 3. Self review pass attaches quotes and demotes unsupported claims.
        if run_self_review:
            with span("self_review", n_sentences=len(draft.sentences)):
                draft = review_and_annotate(draft)
            root.set("grounding_after_review", round(draft.grounding_rate, 3))

        if persist:
            save_draft(draft)
        return draft
