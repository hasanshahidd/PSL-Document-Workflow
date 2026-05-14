"""Vision-model OCR fallback. Delegates to the LLM provider."""
from __future__ import annotations
import json
from dataclasses import dataclass
from providers.registry import get_llm


@dataclass
class VisionResult:
    text: str
    confidence: float


SYSTEM = (
    "You transcribe document images literally. Preserve line breaks. "
    "If a word is unreadable, write [illegible]. "
    "Reply with a JSON object: {\"text\": <transcription>, \"confidence\": <0-1>}."
)


def ocr_image_with_vision(png_bytes: bytes) -> VisionResult:
    resp = get_llm().complete_vision(
        system=SYSTEM, image_png=png_bytes, instruction="Transcribe this page.", max_tokens=4096,
    )
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
        return VisionResult(text=parsed.get("text", ""), confidence=float(parsed.get("confidence", 0.7)))
    except (json.JSONDecodeError, ValueError):
        return VisionResult(text=raw, confidence=0.5)
