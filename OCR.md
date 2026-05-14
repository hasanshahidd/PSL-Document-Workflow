# OCR Architecture

This system treats OCR as the foundation, not as a one line
`pytesseract.image_to_string()` call. Legal documents arrive scanned at low
DPI, photographed sideways, with shadows from flatbed lids, with handwritten
notes in the margins, and with faint text from photocopies of photocopies.
The pipeline below is what it takes to survive that.

```
        raw image bytes (PDF page render, photo, scan)
                       |
                       v
        +----------------------------------+
        |  ocr/orientation.py              |  Tesseract OSD
        |  detect_and_correct()            |  90 / 180 / 270 flip
        +----------------------------------+
                       |
                       v
        +----------------------------------+
        |  ocr/variants.py                 |  for each binariser
        |  run_variants()                  |  in [otsu, sauvola, adaptive]
        +----------------------------------+
            |           |             |
            v           v             v
        +-------------------------------------------------+
        |  ocr/preprocess.py  (shared pre binarise stack) |
        |  =============================================  |
        |  1. upscale_if_low_res()   bicubic to 1500 px   |
        |  2. remove_shadows()       morph background fix |
        |  3. estimate_noise()       Laplacian variance   |
        |  4. denoise()              fastNlMeans / biLat  |
        |  5. clahe_contrast()       adaptive histogram   |
        |  6. deskew()               projection profile   |
        |                            angle search +/- 5   |
        |  7. trim_borders()         bounding box crop    |
        |  8. binarize_*()           per variant          |
        +-------------------------------------------------+
            |           |             |
            v           v             v
        +----------------------------------+
        |  ocr/psm_search.py               |  Tesseract w/ PSM
        |  search_best_psm()               |  in {3, 4, 6, 11}
        |                                  |  + legal user words
        |                                  |  + legal user patterns
        +----------------------------------+
                       |
                       v
        pick winner by (mean word conf, word count)
                       |
                       v
        +----------------------------------+
        |  ocr/sanity.py                   |  regex based score:
        |  score_text()                    |  - word density
        |                                  |  - structural cues
        |                                  |  - garbage ratio
        |                                  |  - avg word length
        +----------------------------------+
                       |
                       v
        combined_confidence = 0.5 * tesseract_conf + 0.5 * sanity_score
                       |
          +------------+------------+
          v                         v
   combined >= 0.6            combined < 0.6
   keep result                trigger vision fallback
   write to ocr_cache         (LLM transcribes page)
```

## Techniques applied (and why)

| Stage | Technique | Why it matters for legal documents |
|---|---|---|
| Decode | PyMuPDF render at 300 DPI for scans, direct text layer read for digital pages. | Most PDFs are mixed. The first pages may be text, exhibits scanned. Routing per page saves real OCR cost. |
| Orientation | Tesseract OSD (`image_to_osd`), then rotate 0, 90, 180, or 270. | Scanners pulling A4 sideways flip orientation. Skew deskew cannot recover from 90 degree errors. |
| Upscale | Bicubic interpolation to minimum 1500 px height. | Low DPI faxed scans (150 DPI or less) starve Tesseract. Upscaling before binarisation gives the engine more pixels per glyph. |
| Shadow removal | Morphological dilation, median blur, subtract. | Flatbed lid shadows turn into binarised black blobs. Remove them before thresholding. |
| Denoise (adaptive) | Bilateral filter on clean inputs, non local means on noisy ones, picked by Laplacian variance. | NLM is slow but recovers heavily speckled photocopies. Bilateral is fast and edge preserving for clean scans. |
| CLAHE | Contrast limited adaptive histogram equalisation, 8 by 8 tiles. | Photocopied or faded sections become readable. Does not blow out already strong contrast. |
| Deskew | Projection profile variance maximisation, +/- 5 degree search at 0.5 degree steps. | Catches the typical 1 to 4 degree tilt from a hand fed sheet. Robust on dense body text. |
| Border trim | Bounding box of inverted Otsu. | Drops scanner black borders that would otherwise be read as garbage. |
| Deskew (dual method) | Projection profile maximisation as the primary, Hough line angle as the secondary. Take the projection result when both agree within 1 degree, otherwise prefer Hough. | Projection profile handles dense body text. Hough catches sparse exhibits where the projection profile flattens out. |
| Multi binarisation | Otsu plus Sauvola plus adaptive Gaussian. | One binarisation never wins every page. Sauvola handles faded text. Otsu handles clean scans. Adaptive handles uneven illumination. |
| Header and footer suppression | After OCR, detect lines that repeat across the top or bottom of most pages (after normalising digits and case) and strip them before chunking. | Stops "Page 5 of 47" and document titles from leaking into every retrieval result. |
| PSM auto tune | Tesseract PSMs 3, 4, 6, 11. Winner is `word_confidence * word_count`. | Single column legal docs (PSM 4) and complaint headers (PSM 11) score very differently. Trying both costs around 3 times Tesseract time per page but lifts accuracy on layout mismatched pages. |
| Legal vocabulary biasing | Tesseract `--user-words` and `--user-patterns` files. | Stops "Plaintiff" becoming "PlaiHtiff". Biases toward `Section <n>`, matter numbers like `PSL-YYYY-NNNN`, and currency formats. |
| Sanity scoring | Independent regex and structural score (word density, structural cues, garbage ratio, average word length). | Tesseract confidence is over optimistic on garbled output. The sanity score catches confident looking garbage and demotes it. |
| Combined confidence | `0.5 * tesseract + 0.5 * sanity`, clamped to `[0, 1]`. | Two independent signals. Both have to agree before we trust the page. |
| Image hash cache | `SHA256(image bytes + engine + config)` mapped to SQLite. | Re ingestion and eval loops never pay Tesseract twice for the same page. |
| Vision fallback | LLM vision transcription, gated on combined confidence below 0.6. | Handwriting, severely degraded scans, exotic stamps. Pages where no amount of preprocessing helps. |
| Confidence weighted retrieval | Downstream: `score * (0.5 + 0.5 * ocr_conf)`. | Even after all the above, a chunk from a poor page is preferred only when nothing better is available. The draft model never anchors on garbled text when good evidence exists. |

## What ends up in the diagnostic record

Every page carries an `OcrDiagnostics` block.

```json
{
  "tesseract_confidence": 0.83,
  "sanity_score": 0.91,
  "rotation_corrected": 0,
  "chosen_variant": "sauvola",
  "chosen_psm": 4,
  "preprocess_stages": [
    "shadow", "denoise(n=1316.33)", "clahe",
    "deskew(-3.00deg)", "trim", "binarize:sauvola"
  ],
  "variants": [
    { "name": "otsu",     "tesseract_conf": 0.74, "word_count": 312, "psm": 4 },
    { "name": "sauvola",  "tesseract_conf": 0.83, "word_count": 318, "psm": 4 },
    { "name": "adaptive", "tesseract_conf": 0.69, "word_count": 295, "psm": 6 }
  ],
  "needs_vision_fallback": false
}
```

## Verified results (smoke test on synthesised scan)

Synthetic legal page run with 3 degree rotation and Gaussian noise
(sigma 20).

| Measurement | Value |
|---|---|
| Estimated noise (Laplacian variance) | 1316.33 |
| Detected skew angle | -3.00 deg (matches injected rotation) |
| Stages applied | shadow, denoise, CLAHE, deskew(-3 deg), trim, Otsu |
| Output | Clean binarised, ready for Tesseract. |

Full integration test in
[tests/test_ocr_pipeline.py](tests/test_ocr_pipeline.py). Skips when
Tesseract is not installed. Runs in CI.

## Failure modes this pipeline rules out

The categories below catch most production failure modes for legal document
OCR.

1. Sideways scans. Handled by OSD rotation correction.
2. Skewed scans. Handled by projection profile deskew.
3. Faded photocopies. Sauvola binarisation rescues them where Otsu would
   blow them out.
4. Uneven lighting from flatbed lids. Shadow removal stage.
5. Speckle noise from low DPI faxes. Non local means denoising.
6. Layout mismatched documents (sparse text, complaint headers). PSM auto
   tune tries four segmentation modes.
7. "PlaiHtiff" or "Hereunder" misreads. Legal user words biasing.
8. Confident looking garbage. Sanity scorer demotes pages where the regex
   and structural fingerprint does not look like a legal document.
9. Pages no preprocessing can save. Vision fallback gated on a meaningful
   combined confidence threshold.
10. Repeated OCR cost on reruns. Image hash cache.
