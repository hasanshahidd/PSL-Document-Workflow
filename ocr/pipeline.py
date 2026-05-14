"""End-to-end OCR pipeline.

Order of operations on a noisy scan:

    decode image bytes
        |
        v
    detect + correct rotation  (Tesseract OSD)
        |
        v
    run 3-variant consensus OCR  (otsu / sauvola / adaptive)
        - each variant goes through: upscale, shadow-removal, denoise,
          CLAHE, deskew, trim, binarise
        - each is OCR'd with PSM auto-tune (PSMs 3/4/6/11)
        - winning variant by mean word-confidence + word count
        |
        v
    legal-vocabulary biasing applied via Tesseract user-words + user-patterns
        |
        v
    post-OCR sanity scoring  (regex/structural)
        |
        v
    combined confidence = mean(tesseract_conf, sanity_score)
        |
        v
    if combined_confidence < vision_fallback_threshold:
        flag for vision fallback (caller's responsibility)

The full diagnostic trace is returned in `OcrPipelineResult.diagnostics`
so it can be exposed via /pages/{doc_id}/{page}/ocr-trace.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from logging_setup import get_logger
from obs.tracer import span
from ocr import cache as ocr_cache
from ocr.orientation import detect_and_correct
from ocr.sanity import score_text, SanityScore
from ocr.variants import VariantRun, run_variants
from ocr.vocabulary import tesseract_config_for_legal


log = get_logger(__name__)


@dataclass
class OcrPipelineResult:
    text: str
    confidence: float                 # final combined score in [0,1]
    tesseract_confidence: float       # raw OCR engine confidence
    sanity: SanityScore
    rotation_corrected: int
    chosen_variant: str
    chosen_psm: int
    preprocess_stages: list[str]
    variants: list[dict] = field(default_factory=list)
    engine: str = "tesseract+legal-vocab"
    needs_vision_fallback: bool = False
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sanity"] = asdict(self.sanity)
        return d


def _decode(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run(image_bytes: bytes,
        confidence_floor: float = 0.6,
        use_cache: bool = True) -> OcrPipelineResult:
    config = tesseract_config_for_legal()
    key = ocr_cache.make_key(image_bytes, "tesseract+pipeline", config)

    if use_cache:
        hit = ocr_cache.get(key)
        if hit is not None:
            return OcrPipelineResult(
                text=hit.text,
                confidence=hit.confidence,
                tesseract_confidence=hit.metadata.get("tesseract_confidence", hit.confidence),
                sanity=SanityScore(**hit.metadata["sanity"]),
                rotation_corrected=hit.metadata.get("rotation_corrected", 0),
                chosen_variant=hit.metadata.get("chosen_variant", "?"),
                chosen_psm=hit.metadata.get("chosen_psm", -1),
                preprocess_stages=hit.metadata.get("preprocess_stages", []),
                variants=hit.metadata.get("variants", []),
                engine=hit.engine,
                needs_vision_fallback=hit.confidence < confidence_floor,
                cached=True,
            )

    with span("ocr.pipeline") as s:
        img = _decode(image_bytes)
        with span("ocr.orientation"):
            img, orient = detect_and_correct(img)
        with span("ocr.variants") as vs:
            winner, runs = run_variants(img, config_extra=config)
            vs.set_many(
                runs=len(runs),
                chosen=winner.name,
                chosen_psm=winner.psm_result.psm,
                tesseract_conf=round(winner.psm_result.confidence, 3),
            )

        text = winner.psm_result.text
        tess_conf = winner.psm_result.confidence
        sanity = score_text(text)
        combined = 0.5 * tess_conf + 0.5 * sanity.score
        combined = round(combined, 3)

        s.set_many(
            rotation=orient.rotation,
            tesseract_conf=round(tess_conf, 3),
            sanity_score=sanity.score,
            combined=combined,
            needs_vision=combined < confidence_floor,
        )

        result = OcrPipelineResult(
            text=text,
            confidence=combined,
            tesseract_confidence=round(tess_conf, 3),
            sanity=sanity,
            rotation_corrected=orient.rotation,
            chosen_variant=winner.name,
            chosen_psm=winner.psm_result.psm,
            preprocess_stages=winner.preprocess_stages,
            variants=[
                {
                    "name": r.name,
                    "tesseract_conf": round(r.psm_result.confidence, 3),
                    "word_count": r.psm_result.word_count,
                    "psm": r.psm_result.psm,
                }
                for r in runs
            ],
            needs_vision_fallback=combined < confidence_floor,
        )

        if use_cache:
            ocr_cache.put(key, ocr_cache.CachedOcr(
                text=text,
                confidence=combined,
                engine="tesseract+pipeline",
                sanity_score=sanity.score,
                metadata={
                    "tesseract_confidence": result.tesseract_confidence,
                    "sanity": asdict(sanity),
                    "rotation_corrected": orient.rotation,
                    "chosen_variant": winner.name,
                    "chosen_psm": winner.psm_result.psm,
                    "preprocess_stages": winner.preprocess_stages,
                    "variants": result.variants,
                },
            ))

        return result
