"""Tesseract PSM (Page Segmentation Mode) auto-tuner.

Different page layouts respond to different PSMs. We try a small set and
pick the result with the highest mean word-confidence — a trick that
typically gains a few percentage points on documents whose layout doesn't
match Tesseract's default assumption.

PSM cheatsheet (we use a subset):
  3 — fully automatic page segmentation (default)
  4 — single column of variable-size text
  6 — single uniform block of text
  11 — sparse text (Find as much text as possible in no particular order)
  12 — sparse text with OSD
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pytesseract


CANDIDATE_PSMS = (3, 4, 6, 11)


@dataclass
class PsmResult:
    text: str
    confidence: float          # mean word confidence in [0,1]
    psm: int
    word_count: int
    word_confidences: list[float]


def _tesseract_run(img: np.ndarray, psm: int, lang: str, config_extra: str) -> PsmResult:
    config = f"--oem 1 --psm {psm} {config_extra}".strip()
    data = pytesseract.image_to_data(img, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confs: list[float] = []
    for word, conf in zip(data["text"], data["conf"]):
        if not word.strip():
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        words.append(word)
        confs.append(c / 100.0)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return PsmResult(
        text=" ".join(words),
        confidence=mean_conf,
        psm=psm,
        word_count=len(words),
        word_confidences=confs,
    )


def search_best_psm(img: np.ndarray, lang: str = "eng",
                    config_extra: str = "",
                    candidates=CANDIDATE_PSMS) -> tuple[PsmResult, list[PsmResult]]:
    """Run Tesseract under every candidate PSM and return the winner.

    'Winner' = highest mean word-confidence with a minimum word count. If
    everything is empty, fall back to the result with the most words.
    """
    runs: list[PsmResult] = []
    for psm in candidates:
        try:
            runs.append(_tesseract_run(img, psm, lang, config_extra))
        except pytesseract.TesseractError:
            continue
    if not runs:
        return PsmResult("", 0.0, -1, 0, []), []
    runs.sort(key=lambda r: (r.word_count >= 5, r.confidence, r.word_count), reverse=True)
    return runs[0], runs
