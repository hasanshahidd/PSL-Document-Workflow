"""Detect and strip repeating headers and footers across pages.

Scanned legal documents typically reuse the same header line ("Confidential
- Pearson Specter Litt", a case number, a date stamp) and footer ("Page 3
of 47") on every page. These leak into chunks, dilute embedding similarity
on real content, and clutter retrieval results.

This module runs after OCR but before chunking. It compares the first and
last few lines of every page. Lines that appear on a majority of pages get
suppressed. A line is "the same" if its normalised form (lowercase, digits
collapsed, whitespace squeezed) matches.
"""
from __future__ import annotations
import re
from collections import Counter
from schemas import PageContent


_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _normalise(line: str) -> str:
    s = line.strip().lower()
    s = _DIGIT_RE.sub("N", s)  # "page 3 of 47" -> "page N of N"
    s = _WS_RE.sub(" ", s)
    return s


def _candidate_lines(text: str, n_top: int = 3, n_bot: int = 3) -> tuple[list[str], list[str]]:
    raw_lines = [l for l in text.splitlines() if l.strip()]
    top = raw_lines[:n_top]
    bot = raw_lines[-n_bot:] if len(raw_lines) > n_bot else []
    return top, bot


def suppress_repeating(pages: list[PageContent],
                       min_frequency: float = 0.5,
                       min_pages: int = 3) -> list[PageContent]:
    """Return a new list of pages with repeating header/footer lines removed.

    A line is dropped from a page when:
      - the document has at least `min_pages` pages, and
      - the normalised form of the line appears in the top or bottom band on
        at least `min_frequency` fraction of pages.
    """
    if len(pages) < min_pages:
        return pages

    top_counts: Counter[str] = Counter()
    bot_counts: Counter[str] = Counter()
    for p in pages:
        top, bot = _candidate_lines(p.text)
        top_counts.update({_normalise(l) for l in top})
        bot_counts.update({_normalise(l) for l in bot})

    threshold = max(min_pages, int(len(pages) * min_frequency))
    headers = {k for k, n in top_counts.items() if n >= threshold and k}
    footers = {k for k, n in bot_counts.items() if n >= threshold and k}

    if not headers and not footers:
        return pages

    cleaned: list[PageContent] = []
    for p in pages:
        lines = p.text.splitlines()
        kept: list[str] = []
        # strip matching lines only from the actual top / bottom bands,
        # never from the middle of the page
        head_band_remaining = 3
        for l in lines:
            if head_band_remaining > 0 and _normalise(l) in headers:
                head_band_remaining -= 1
                continue
            kept.append(l)
            if l.strip():
                head_band_remaining = 0
        # now do the same from the tail
        kept_rev = list(reversed(kept))
        tail_band_remaining = 3
        kept_final_rev: list[str] = []
        for l in kept_rev:
            if tail_band_remaining > 0 and _normalise(l) in footers:
                tail_band_remaining -= 1
                continue
            kept_final_rev.append(l)
            if l.strip():
                tail_band_remaining = 0
        cleaned_text = "\n".join(reversed(kept_final_rev)).strip()
        cleaned.append(p.model_copy(update={"text": cleaned_text}))

    return cleaned
