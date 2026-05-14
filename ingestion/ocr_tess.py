"""Tesseract shim — kept for backwards compatibility. Delegates to the OCR provider."""
from __future__ import annotations
from dataclasses import dataclass
from providers.registry import get_ocr


@dataclass
class OcrResult:
    text: str
    confidence: float


def ocr_image_bytes(png_bytes: bytes) -> OcrResult:
    r = get_ocr().ocr(png_bytes)
    return OcrResult(text=r.text, confidence=r.confidence)
