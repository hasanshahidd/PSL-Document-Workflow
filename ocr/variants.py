"""Multi-variant consensus OCR.

We OCR several pre-processed variants of the same page (different
binarisers, with/without shadow removal, etc.) and pick the highest-
confidence run. This costs a few extra Tesseract calls but is the cheapest
way to recover documents where one preprocessing choice happens to
catastrophically fail (e.g. Otsu binarisation crushes faded text but
Sauvola handles it).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ocr.preprocess import (
    PreprocessReport, binarize_adaptive, binarize_otsu, binarize_sauvola,
    clahe_contrast, denoise, deskew, remove_shadows, trim_borders,
    upscale_if_low_res, _ensure_gray, estimate_noise,
)
from ocr.psm_search import PsmResult, search_best_psm


@dataclass
class VariantRun:
    name: str
    psm_result: PsmResult
    preprocess_stages: list[str] = field(default_factory=list)


def _common_clean(img: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stages: list[str] = []
    img = _ensure_gray(img)
    img, up = upscale_if_low_res(img)
    if up:
        stages.append("upscale")
    img = remove_shadows(img); stages.append("shadow")
    noise = estimate_noise(img)
    img = denoise(img, noise_score=noise); stages.append(f"denoise(n={noise:.1f})")
    img = clahe_contrast(img); stages.append("clahe")
    img, angle = deskew(img)
    if angle != 0.0:
        stages.append(f"deskew({angle:.2f}deg)")
    img = trim_borders(img); stages.append("trim")
    return img, stages


def run_variants(img: np.ndarray, lang: str = "eng",
                 config_extra: str = "") -> tuple[VariantRun, list[VariantRun]]:
    """Run three binariser variants. Return (winner, all_runs)."""
    cleaned, common_stages = _common_clean(img)

    variants: list[tuple[str, np.ndarray]] = [
        ("otsu", binarize_otsu(cleaned)),
        ("sauvola", binarize_sauvola(cleaned)),
        ("adaptive", binarize_adaptive(cleaned)),
    ]

    runs: list[VariantRun] = []
    for name, var in variants:
        psm_result, _ = search_best_psm(var, lang=lang, config_extra=config_extra)
        runs.append(VariantRun(
            name=name,
            psm_result=psm_result,
            preprocess_stages=common_stages + [f"binarize:{name}"],
        ))

    # Winner: highest confidence with at least 5 words. Otherwise: most words.
    runs.sort(key=lambda r: (r.psm_result.word_count >= 5,
                             r.psm_result.confidence,
                             r.psm_result.word_count),
              reverse=True)
    return runs[0], runs
