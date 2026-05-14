from edits.rules import ingest_signals, top_rules, MIN_SUPPORT_FOR_INJECTION


def test_rule_support_increments_on_repeat():
    ingest_signals([{"category": "tone_shift", "rule": "use active voice"}], doc_type="contract")
    ingest_signals([{"category": "tone_shift", "rule": "use active voice"}], doc_type="contract")
    rules = top_rules(doc_type="contract")
    assert any(r["rule"] == "use active voice" and r["support"] >= MIN_SUPPORT_FOR_INJECTION for r in rules)


def test_low_support_rules_are_not_returned():
    ingest_signals([{"category": "other", "rule": "rare rule"}])
    assert not any(r["rule"] == "rare rule" for r in top_rules())


def test_signals_without_rule_text_are_ignored():
    n = ingest_signals([{"category": "other", "rule": None}])
    assert n == 0
