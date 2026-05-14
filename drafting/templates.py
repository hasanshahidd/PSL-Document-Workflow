"""Draft templates.

Each template is a fixed set of sections plus the writing instructions
appropriate for that document family. The legacy "Case Fact Summary"
template is preserved for litigation style inputs. A "Technical Document
Summary" template was added for standards, benchmarks, reports, manuals,
and similar non litigation documents where Parties and Timeline of Events
do not apply.

The right template for a given draft is chosen by `select_template`
which inspects the `doc_type` extracted during ingestion.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftTemplate:
    name: str                       # machine id stored on the Draft
    title: str                      # human title used in narration
    sections: tuple[str, ...]       # long section labels passed to the planner
    headings: dict[str, str]        # long label to short markdown heading
    matches_doc_types: tuple[str, ...]
    section_system: str             # prompt used by per_section.generate_section


_CASE_SECTION_SYSTEM = """You write ONE section of a Case Fact Summary, grounded strictly in the
provided evidence.

RULES. Non negotiable.

1. Every factual sentence MUST end with one or more citations of the form
   [chunk_id]. You may ONLY cite chunk_ids that appear in the ALLOWED
   CITATIONS list at the top of the user message.
2. If the allowed evidence does not support this section, write exactly:
   "No supporting evidence found in the provided documents."
3. Do NOT invent names, dates, amounts, or quotes.
4. Quote sparingly. Paraphrase the evidence faithfully.
5. Output ONLY the body sentences of this section. Do NOT include the
   section heading. The orchestrator adds it.
"""


_TECH_SECTION_SYSTEM = """You write ONE section of a Technical Document Summary, grounded strictly
in the provided evidence.

RULES. Non negotiable.

1. Every factual sentence MUST end with one or more citations of the form
   [chunk_id]. You may ONLY cite chunk_ids that appear in the ALLOWED
   CITATIONS list at the top of the user message.
2. If the allowed evidence does not support this section, write exactly:
   "No supporting evidence found in the provided documents."
3. Do NOT invent version numbers, dates, vendor names, or capabilities.
4. Prefer concrete technical facts (versions, dates, IDs, named modules,
   numeric thresholds) over general framing language.
5. Output ONLY the body sentences of this section. Do NOT include the
   section heading. The orchestrator adds it.
"""


CASE_FACT_SUMMARY = DraftTemplate(
    name="case_fact_summary",
    title="Case Fact Summary",
    sections=(
        "Parties involved and their roles",
        "Key dates and timeline of events",
        "Subject matter and underlying transaction or dispute",
        "Material facts established in the documents",
        "Outstanding questions or facts that are unclear",
    ),
    headings={
        "Parties involved and their roles": "Parties",
        "Key dates and timeline of events": "Timeline",
        "Subject matter and underlying transaction or dispute": "Subject Matter",
        "Material facts established in the documents": "Material Facts",
        "Outstanding questions or facts that are unclear": "Uncertain or Unclear",
    },
    matches_doc_types=(
        "contract", "agreement", "notice", "complaint", "affidavit",
        "title_report", "memo", "subpoena", "stipulation", "deposition",
        "indictment", "injunction", "pleading", "amendment", "addendum",
    ),
    section_system=_CASE_SECTION_SYSTEM,
)


TECHNICAL_DOC_SUMMARY = DraftTemplate(
    name="technical_document_summary",
    title="Technical Document Summary",
    sections=(
        "Document type, version, and authoring body",
        "Scope and intended audience",
        "Key recommendations or technical guidance",
        "Timeline of updates or revisions",
        "Out of scope items or unclear areas",
    ),
    headings={
        "Document type, version, and authoring body": "Document",
        "Scope and intended audience": "Scope and Audience",
        "Key recommendations or technical guidance": "Key Recommendations",
        "Timeline of updates or revisions": "Timeline of Updates",
        "Out of scope items or unclear areas": "Out of Scope or Unclear",
    },
    matches_doc_types=(
        "benchmark", "standard", "manual", "guide", "report",
        "specification", "policy", "whitepaper", "rfc", "handbook",
        "playbook", "runbook", "framework",
    ),
    section_system=_TECH_SECTION_SYSTEM,
)


TEMPLATES: tuple[DraftTemplate, ...] = (CASE_FACT_SUMMARY, TECHNICAL_DOC_SUMMARY)


def select_template(doc_type: str | None) -> DraftTemplate:
    """Pick the right template for a doc_type. Falls back to Case Fact Summary
    when the type is unknown or unrecognised so legal style docs keep their
    existing behaviour.
    """
    if not doc_type:
        return CASE_FACT_SUMMARY
    needle = doc_type.lower().strip()
    for tpl in TEMPLATES:
        if needle in tpl.matches_doc_types:
            return tpl
    return CASE_FACT_SUMMARY
