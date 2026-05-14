"""Sentence-level alignment of an original draft vs. operator-edited text.

We compare sentence lists with difflib and emit one record per operation:
kept, changed (replace), added (insert), removed (delete). Sentence
boundaries reuse the citer's splitter so headings are preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
from drafting.citer import _split_sentences


@dataclass
class EditOp:
    op: str  # "kept" | "changed" | "added" | "removed"
    original: str | None
    edited: str | None


def align(original_text: str, edited_text: str) -> list[EditOp]:
    a = _split_sentences(original_text)
    b = _split_sentences(edited_text)
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    ops: list[EditOp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ops.append(EditOp("kept", a[i1 + k], b[j1 + k]))
        elif tag == "replace":
            # zip what we can, then handle leftovers as add/remove
            common = min(i2 - i1, j2 - j1)
            for k in range(common):
                ops.append(EditOp("changed", a[i1 + k], b[j1 + k]))
            for k in range(common, i2 - i1):
                ops.append(EditOp("removed", a[i1 + k], None))
            for k in range(common, j2 - j1):
                ops.append(EditOp("added", None, b[j1 + k]))
        elif tag == "delete":
            for k in range(i1, i2):
                ops.append(EditOp("removed", a[k], None))
        elif tag == "insert":
            for k in range(j1, j2):
                ops.append(EditOp("added", None, b[k]))
    return ops


def summary(ops: list[EditOp]) -> dict:
    counts = {"kept": 0, "changed": 0, "added": 0, "removed": 0}
    for op in ops:
        counts[op.op] += 1
    return counts
