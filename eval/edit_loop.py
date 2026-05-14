"""Measure improvement over rounds of (generate -> simulate edit -> capture).

If the feedback loop is working, the operator's edit distance should trend
down across rounds: later drafts already reflect learned style, so there is
less for the simulated operator to change.
"""
from __future__ import annotations
from difflib import SequenceMatcher
from drafting.generator import generate_case_fact_summary
from edits.capture import capture_edit
from drafting.citer import _split_sentences
from eval.simulate_edits import simulate_edit


def edit_distance_ratio(a: str, b: str) -> float:
    """1 - SequenceMatcher ratio over sentence sequences. Lower = closer."""
    a_s = _split_sentences(a)
    b_s = _split_sentences(b)
    if not a_s and not b_s:
        return 0.0
    return 1.0 - SequenceMatcher(a=a_s, b=b_s, autojunk=False).ratio()


def run_loop(doc_ids: list[str], rounds: int = 5) -> dict:
    history = []
    for r in range(1, rounds + 1):
        draft = generate_case_fact_summary(doc_ids)
        edited = simulate_edit(draft.text)
        dist = edit_distance_ratio(draft.text, edited)
        capture_edit(draft.draft_id, edited)
        history.append(
            {
                "round": r,
                "draft_id": draft.draft_id,
                "edit_distance": round(dist, 4),
                "n_sentences": len(draft.sentences),
                "grounding_rate": round(draft.grounding_rate, 4),
            }
        )

    first = history[0]["edit_distance"]
    last = history[-1]["edit_distance"]
    return {
        "rounds": history,
        "first_distance": first,
        "last_distance": last,
        "improvement": round(first - last, 4),
        "improved": last < first,
    }
