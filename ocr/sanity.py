"""Post-OCR sanity scoring.

Tesseract's word-confidence is a useful signal but not a sufficient one —
on a really garbled page Tesseract can be 'confident' that it sees words
that aren't actually words. This module computes an independent sanity
score from the OCR text itself, based on signals that legal documents
should exhibit:

  - high ratio of dictionary-shaped tokens (letters, plausibly Latin)
  - presence of structural cues (digits, currency symbols, capitalised
    proper nouns, section markers)
  - low ratio of garbage characters (non-printable, weird Unicode)
  - reasonable average word length (1.5 - 12 chars)

The final score is a weighted blend, clamped to [0, 1]. We treat this as
an independent estimate of OCR quality and combine it with Tesseract's
mean word-confidence before triggering the vision fallback.
"""
from __future__ import annotations
import re
import string
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
DIGIT_RE = re.compile(r"\d")
CURRENCY_RE = re.compile(r"[$€£¥]|\bUSD\b|\bEUR\b|\bdollars?\b", re.IGNORECASE)
DATE_HINT_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"\d{1,2}[/\-]\d{1,2}|\d{4})\b",
    re.IGNORECASE,
)
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
SECTION_HINT_RE = re.compile(
    r"\b(?:Section|Article|Clause|Whereas|Party|Parties|Agreement|"
    r"Plaintiff|Defendant|Court|Notice|Hereby|Pursuant)\b",
    re.IGNORECASE,
)


@dataclass
class SanityScore:
    score: float
    word_count: int
    word_density: float
    garbage_ratio: float
    has_digits: bool
    has_currency: bool
    has_dates: bool
    has_section_terms: bool
    avg_word_length: float


_PRINTABLE = set(string.printable + "“”‘’–—…•")


def score_text(text: str) -> SanityScore:
    if not text:
        return SanityScore(0.0, 0, 0.0, 1.0, False, False, False, False, 0.0)

    tokens = text.split()
    word_count = len(tokens)
    if word_count == 0:
        return SanityScore(0.0, 0, 0.0, 1.0, False, False, False, False, 0.0)

    word_like = WORD_RE.findall(text)
    word_density = len(word_like) / max(word_count, 1)

    total_chars = len(text)
    garbage = sum(1 for c in text if c not in _PRINTABLE and not c.isspace())
    garbage_ratio = garbage / max(total_chars, 1)

    has_digits = bool(DIGIT_RE.search(text))
    has_currency = bool(CURRENCY_RE.search(text))
    has_dates = bool(DATE_HINT_RE.search(text))
    has_section_terms = bool(SECTION_HINT_RE.search(text))

    avg_len = sum(len(w) for w in word_like) / max(len(word_like), 1)

    # length penalty: very short or very long average word length is suspicious
    length_score = 1.0 - min(abs(avg_len - 5.0) / 5.0, 1.0)

    structural_score = sum([has_digits, has_currency, has_dates, has_section_terms]) / 4.0

    score = (
        0.45 * word_density +
        0.30 * length_score +
        0.20 * structural_score +
        0.05 * (1.0 - min(garbage_ratio * 5, 1.0))
    )
    score = max(0.0, min(1.0, score))

    return SanityScore(
        score=round(score, 3),
        word_count=word_count,
        word_density=round(word_density, 3),
        garbage_ratio=round(garbage_ratio, 3),
        has_digits=has_digits,
        has_currency=has_currency,
        has_dates=has_dates,
        has_section_terms=has_section_terms,
        avg_word_length=round(avg_len, 2),
    )
