"""End-to-end ingestion CLI:  python -m ingestion.run <file>

Routes each page, runs the rich OCR pipeline on scans (preprocessing,
multi-variant consensus, PSM autotune, sanity scoring), and falls back
to the vision model only when the combined confidence is below threshold.
The result is a ProcessedDocument JSON written under data/processed/.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path
from config import settings
from schemas import OcrDiagnostics, PageContent, ProcessedDocument
from ingestion.pdf_router import route_pdf, route_image
from ingestion.ocr_vision import ocr_image_with_vision
from ingestion.struct_extract import extract_fields
from ocr.header_footer import suppress_repeating
from providers.registry import get_ocr


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}_{h}"


def _ocr_with_diagnostics(image_png: bytes) -> tuple[str, float, OcrDiagnostics]:
    """Use the provider's full pipeline if available; otherwise the simple ocr()."""
    ocr_provider = get_ocr()
    if hasattr(ocr_provider, "ocr_full"):
        full = ocr_provider.ocr_full(image_png)
        diag = OcrDiagnostics(
            tesseract_confidence=full.tesseract_confidence,
            sanity_score=full.sanity.score,
            rotation_corrected=full.rotation_corrected,
            chosen_variant=full.chosen_variant,
            chosen_psm=full.chosen_psm,
            preprocess_stages=full.preprocess_stages,
            variants=full.variants,
            needs_vision_fallback=full.needs_vision_fallback,
        )
        return full.text, full.confidence, diag
    # fallback for fakes
    result = ocr_provider.ocr(image_png)
    return result.text, result.confidence, OcrDiagnostics()


def process(path: Path, use_vision_fallback: bool = True) -> ProcessedDocument:
    if path.suffix.lower() == ".pdf":
        routes = route_pdf(path)
    elif path.suffix.lower() in IMAGE_EXTS:
        routes = route_image(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")

    pages: list[PageContent] = []
    for r in routes:
        if r.route == "digital":
            pages.append(PageContent(
                page=r.page, text=r.digital_text, ocr_confidence=1.0,
                source="digital",
            ))
            continue

        text, conf, diag = _ocr_with_diagnostics(r.image_png)
        source: str = "tesseract"

        if (use_vision_fallback
                and conf < settings.ocr_confidence_threshold
                and settings.anthropic_api_key):
            try:
                v = ocr_image_with_vision(r.image_png)
                if v.confidence > conf:
                    text, conf, source = v.text, v.confidence, "vision"
            except Exception as e:
                print(f"[warn] vision fallback failed on page {r.page}: {e}",
                      file=sys.stderr)

        pages.append(PageContent(
            page=r.page, text=text, ocr_confidence=conf, source=source,
            ocr_diagnostics=diag,
        ))

    # Strip repeating headers and footers across pages before chunking.
    pages = suppress_repeating(pages)

    full_text = "\n\n".join(p.text for p in pages)
    fields = extract_fields(full_text)
    return ProcessedDocument(
        doc_id=_doc_id(path), filename=path.name, pages=pages,
        fields=fields, full_text=full_text,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a legal-style document.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable vision fallback.")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 1

    doc = process(args.path, use_vision_fallback=not args.no_vision)
    out_path = settings.processed_dir / f"{doc.doc_id}.json"
    out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  pages: {len(doc.pages)}  mean_conf: {doc.mean_confidence:.2f}  "
          f"doc_type: {doc.fields.doc_type}")
    for p in doc.pages:
        if p.ocr_diagnostics:
            d = p.ocr_diagnostics
            stages = ", ".join(d.preprocess_stages[:6])
            print(f"  page {p.page}: tess={d.tesseract_confidence:.2f} "
                  f"sanity={d.sanity_score:.2f} variant={d.chosen_variant} "
                  f"psm={d.chosen_psm} stages=[{stages}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
