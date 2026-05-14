"""Integration test for the OCR pipeline.

We synthesise a 'bad scan': legal text, rotated, with noise and a
shadow gradient, then check that the full pipeline:
  - detects + corrects rotation
  - returns text that the sanity scorer recognises as legal
  - produces a combined confidence above a reasonable floor
"""
from __future__ import annotations
import io
import shutil
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
pytesseract = pytest.importorskip("pytesseract")

if not shutil.which("tesseract"):
    pytest.skip("tesseract binary not available", allow_module_level=True)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ocr.pipeline import run  # noqa: E402
from ocr.sanity import score_text  # noqa: E402


LEGAL_TEXT = (
    "LICENSING AGREEMENT\n\n"
    "This Licensing Agreement is entered into on 15 January 2026 by and "
    "between Acme Industries (the Licensor) and Beta Holdings (the "
    "Licensee).\n\n"
    "Section 1. Subject Matter. The Licensor grants the Licensee a "
    "non-exclusive license to use the proprietary software known as "
    "Helios v3 for internal business purposes.\n\n"
    "Section 2. Consideration. The Licensee shall pay the Licensor the "
    "sum of two hundred and fifty thousand US dollars ($250,000), payable "
    "within thirty days of execution.\n\n"
    "Section 3. Termination. Either party may terminate this agreement "
    "upon sixty days prior written notice for any reason."
)


def _render_to_png(text: str, width: int = 1100) -> bytes:
    img = Image.new("RGB", (width, 1500), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((50, 80), text, fill="black", font=font, spacing=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _degrade(png_bytes: bytes) -> bytes:
    arr = np.array(Image.open(io.BytesIO(png_bytes)))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # 2-degree rotation to exercise deskew
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    bgr = cv2.warpAffine(bgr, M, (w, h), borderValue=(255, 255, 255))
    # add Gaussian noise
    noise = np.random.normal(0, 12, bgr.shape).astype(np.int16)
    bgr = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # add a soft diagonal shadow
    shadow = np.linspace(0, 60, w, dtype=np.int16).reshape(1, w, 1)
    shadow = np.tile(shadow, (h, 1, 3))
    bgr = np.clip(bgr.astype(np.int16) - shadow, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def test_pipeline_reads_legal_text_through_noise_and_skew():
    rendered = _render_to_png(LEGAL_TEXT)
    degraded = _degrade(rendered)

    result = run(degraded, use_cache=False)

    # rotation will be small (<5 deg) so OSD reports 0; deskew handles it.
    assert result.chosen_variant in {"otsu", "sauvola", "adaptive"}
    assert result.chosen_psm in {3, 4, 6, 11}
    assert any(stage.startswith("denoise") for stage in result.preprocess_stages)
    assert "clahe" in result.preprocess_stages
    # the sanity scorer should be happy with the recovered text
    s = score_text(result.text)
    assert s.has_section_terms is True
    # OCR may drop or alter a couple of words; accept any clear monetary signal.
    text_lc = result.text.lower()
    assert (
        s.has_currency is True
        or "dollars" in text_lc
        or "thousand" in text_lc
        or "$" in text_lc
        or "250" in text_lc
    ), f"no monetary signal in: {result.text[:200]}"
    assert s.word_density > 0.6
    # combined confidence should be respectable even after degradation
    assert result.confidence > 0.5, result.text[:200]


def test_sanity_score_marks_garbage_text_low():
    s = score_text("!@#$ %^^&* ()_+ ~~~ ::: ;;; xkcd zyxw 999999")
    # garbage scores well below legal-text scores (~0.9). Generous bound
    # since 'xkcd zyxw' still look like words to a regex.
    assert s.score < 0.55


def test_sanity_score_marks_legal_text_high():
    s = score_text(LEGAL_TEXT)
    assert s.score > 0.6
    assert s.has_section_terms is True
    assert s.has_dates is True


def test_sanity_score_separates_legal_from_garbage():
    legal = score_text(LEGAL_TEXT).score
    garbage = score_text("!@#$ %^^&* ()_+ ~~~ ::: ;;; xkcd zyxw 999999").score
    assert legal > garbage + 0.3
