"""Decide per page whether to use the embedded text layer or fall back to OCR.

A PDF page is treated as 'digital' when its text layer contains enough characters
to be useful; otherwise we treat it as scanned and route it to OCR. The image
bytes are rendered once and reused by downstream OCR modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF


DIGITAL_MIN_CHARS = 50  # below this we assume the page is scanned


@dataclass
class PageRoute:
    page: int
    route: str  # "digital" or "scan"
    digital_text: str
    image_png: bytes


def route_pdf(path: Path, dpi: int = 300) -> list[PageRoute]:
    """Open a PDF and classify each page as digital or scan, rendering image bytes."""
    routes: list[PageRoute] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_bytes = pix.tobytes("png")
            is_digital = len(text) >= DIGITAL_MIN_CHARS
            routes.append(
                PageRoute(
                    page=i + 1,
                    route="digital" if is_digital else "scan",
                    digital_text=text,
                    image_png=img_bytes,
                )
            )
    return routes


def route_image(path: Path) -> list[PageRoute]:
    """A standalone image is always treated as a scan."""
    return [
        PageRoute(
            page=1,
            route="scan",
            digital_text="",
            image_png=path.read_bytes(),
        )
    ]
