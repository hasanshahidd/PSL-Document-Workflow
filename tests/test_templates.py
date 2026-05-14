"""Template selection by doc_type.

The legal Case Fact Summary is the safe fallback. Specific known types map
to their family's template.
"""
from drafting.templates import (
    CASE_FACT_SUMMARY, TECHNICAL_DOC_SUMMARY, select_template,
)


def test_unknown_doc_type_falls_back_to_case_fact_summary():
    assert select_template(None) is CASE_FACT_SUMMARY
    assert select_template("") is CASE_FACT_SUMMARY
    assert select_template("unknown") is CASE_FACT_SUMMARY


def test_legal_doc_types_pick_case_fact_summary():
    for t in ["contract", "notice", "complaint", "affidavit", "agreement"]:
        assert select_template(t) is CASE_FACT_SUMMARY, t


def test_technical_doc_types_pick_technical_summary():
    for t in ["benchmark", "standard", "manual", "guide", "report",
              "specification", "policy", "whitepaper", "rfc"]:
        assert select_template(t) is TECHNICAL_DOC_SUMMARY, t


def test_doc_type_matching_is_case_insensitive_and_trimmed():
    assert select_template("  BENCHMARK ") is TECHNICAL_DOC_SUMMARY
    assert select_template("Contract") is CASE_FACT_SUMMARY


def test_templates_have_consistent_sections_and_headings():
    for tpl in (CASE_FACT_SUMMARY, TECHNICAL_DOC_SUMMARY):
        assert len(tpl.sections) == 5, tpl.name
        # every section must have a heading mapping
        for s in tpl.sections:
            assert s in tpl.headings, f"{tpl.name} missing heading for {s}"
        assert tpl.section_system.strip(), tpl.name
