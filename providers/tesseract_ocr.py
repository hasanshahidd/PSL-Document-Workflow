"""Tesseract OCR provider — now backed by the industrial-grade pipeline.

The pipeline returns much richer information than the original word-list
mean confidence: rotation detection, multi-variant binarisation, PSM
auto-tune, post-OCR sanity score, and per-variant diagnostics. We expose
the legacy `ocr(image_png) -> OCRResult` shape on the provider base class
but also publish the full result via `ocr_full()` for callers that want
to inspect or persist diagnostics.
"""
from __future__ import annotations
import pytesseract
from config import settings
from errors import OCRError
from obs.tracer import span
from providers.base import OCRProvider, OCRResult
from ocr.pipeline import OcrPipelineResult, run as run_pipeline


class TesseractOCR(OCRProvider):
    def __init__(self, binary_path: str | None = None):
        path = binary_path or settings.tesseract_cmd
        if path:
            pytesseract.pytesseract.tesseract_cmd = path

    def ocr(self, image_png: bytes) -> OCRResult:
        with span("ocr.tesseract") as s:
            try:
                result = run_pipeline(image_png)
            except Exception as e:
                s.status = "error"
                s.error = f"{type(e).__name__}: {e}"
                raise OCRError(str(e)) from e
            s.set_many(
                tess_conf=result.tesseract_confidence,
                sanity=result.sanity.score,
                combined=result.confidence,
                cached=result.cached,
            )
            return OCRResult(text=result.text, confidence=result.confidence)

    def ocr_full(self, image_png: bytes) -> OcrPipelineResult:
        """Same pipeline but returns the rich diagnostic result."""
        return run_pipeline(image_png)
