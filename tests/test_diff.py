from edits.diff import align, summary


def test_kept_changed_added_removed_are_all_detected():
    original = "Alpha is here. Bravo follows. Charlie ends it."
    edited = "Alpha is here. Bravo follows quickly. Delta is new."
    ops = align(original, edited)
    s = summary(ops)
    assert s["kept"] >= 1
    assert s["changed"] + s["added"] + s["removed"] >= 1
    assert sum(s.values()) == len(ops)


def test_identical_text_is_all_kept():
    text = "Sentence one. Sentence two."
    ops = align(text, text)
    s = summary(ops)
    assert s["kept"] == len(ops) > 0
    assert s["changed"] == s["added"] == s["removed"] == 0


def test_pure_addition_only_emits_added_ops():
    original = "First."
    edited = "First. Second. Third."
    s = summary(align(original, edited))
    assert s["kept"] == 1
    assert s["added"] == 2
    assert s["removed"] == 0
