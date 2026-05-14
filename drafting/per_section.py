"""Per section generation with focused evidence.

Each section is generated independently:
  1. Plan one or two retrieval queries for this section.
  2. Retrieve top K candidates via the bi encoder.
  3. Rerank with the cross encoder. Keep top N.
  4. Ask the LLM for ONLY this section's content, with ONLY these N chunks
     as allowed citations.

The exact section list, headings, and writing instructions come from the
`DraftTemplate` passed in. See `drafting/templates.py`.
"""
from __future__ import annotations
import json
from logging_setup import get_logger
from obs.tracer import span
from providers.registry import get_llm
from retrieval.reranker import rerank
from retrieval.retriever import retrieve
from schemas import EvidenceChunk
from drafting.prompt import PLANNER_SYSTEM, format_evidence
from drafting.templates import DraftTemplate


log = get_logger(__name__)


def _plan_queries(section: str, doc_ids: list[str]) -> list[str]:
    user = f"Section: {section}\nSource document ids: {doc_ids}"
    try:
        resp = get_llm().complete(
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user}],
            max_tokens=192,
        )
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        return json.loads(raw).get("queries", [section])
    except Exception:
        return [section]


def _retrieve_for_section(
    section: str, doc_ids: list[str],
    k_initial: int = 12, k_final: int = 4,
) -> list[EvidenceChunk]:
    queries = _plan_queries(section, doc_ids)
    pool: dict[str, EvidenceChunk] = {}
    for q in queries:
        for hit in retrieve(q, k=k_initial, doc_ids=doc_ids):
            prev = pool.get(hit.chunk_id)
            if prev is None or hit.score > prev.score:
                pool[hit.chunk_id] = hit
    if not pool:
        return []
    candidates = list(pool.values())
    return rerank(queries[0], candidates, top_n=k_final)


def _section_user_message(section_label: str, chunks: list[EvidenceChunk]) -> str:
    allowed = "\n".join(f"  - {c.chunk_id}" for c in chunks) or "  (none)"
    return (
        f"Section to write: {section_label}\n\n"
        "ALLOWED CITATIONS. You may cite ONLY these chunk_ids.\n"
        f"{allowed}\n\n"
        "=====\n\n"
        "Evidence chunks follow.\n\n"
        f"{format_evidence(chunks)}\n\n"
        f"Write the {section_label} section body now."
    )


def generate_section(
    section_label: str,
    doc_ids: list[str],
    template: DraftTemplate,
    k_initial: int = 12,
    k_final: int = 4,
    extra_system: str = "",
) -> tuple[str, list[EvidenceChunk]]:
    """Return (body_markdown, evidence_used_for_this_section).

    `template` carries the writing instructions appropriate for the
    document family this draft belongs to.
    """
    with span("per_section.generate", section=section_label, template=template.name) as s:
        evidence = _retrieve_for_section(section_label, doc_ids, k_initial, k_final)
        s.set_many(n_evidence=len(evidence))
        if not evidence:
            return ("No supporting evidence found in the provided documents.", [])
        system = template.section_system + (("\n\n" + extra_system) if extra_system else "")
        resp = get_llm().complete(
            system=system,
            messages=[{"role": "user", "content": _section_user_message(section_label, evidence)}],
            max_tokens=512,
        )
        return (resp.text.strip(), evidence)


def render_section(heading: str, body: str) -> str:
    return f"## {heading}\n{body.strip()}"


def section_label_to_heading(template: DraftTemplate, section: str) -> str:
    return template.headings.get(section, section)
