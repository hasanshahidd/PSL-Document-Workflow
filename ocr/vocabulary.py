"""Legal-vocabulary biasing for Tesseract.

Tesseract supports `user-words` (a list of words it should be more willing
to commit to) and `user-patterns` (regex-like patterns for formatted
tokens such as Matter numbers). Loading the right vocabulary stops the
engine from turning 'Plaintiff' into 'PlaiHtiff' on faint scans.

The file is generated lazily on first use and cached under data/.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from config import settings


LEGAL_TERMS = [
    # parties / roles
    "Plaintiff", "Defendant", "Petitioner", "Respondent", "Appellant",
    "Appellee", "Licensor", "Licensee", "Lessor", "Lessee", "Grantor",
    "Grantee", "Mortgagor", "Mortgagee", "Trustee", "Beneficiary",
    "Assignor", "Assignee",
    # document types
    "Agreement", "Contract", "Amendment", "Addendum", "Affidavit",
    "Complaint", "Counterclaim", "Memorandum", "Notice", "Stipulation",
    "Subpoena", "Injunction", "Indictment", "Deposition", "Pleading",
    # boilerplate
    "Whereas", "Hereby", "Hereinafter", "Heretofore", "Herein",
    "Pursuant", "Notwithstanding", "Furthermore", "Therefore",
    "Provided", "Foregoing", "Aforementioned",
    # action verbs
    "covenants", "agrees", "warrants", "represents", "acknowledges",
    "stipulates", "alleges", "demands",
    # legal nouns
    "consideration", "indemnification", "confidentiality", "jurisdiction",
    "arbitration", "remedies", "damages", "obligation", "liability",
    "warranty", "covenant", "termination", "breach", "default", "notice",
    "execution", "delivery", "performance",
    # numbers / time
    "thirty", "sixty", "ninety", "hundred", "thousand", "million",
    "calendar", "fiscal", "quarterly", "annual", "annually",
]


# Tesseract user-pattern syntax: \n digit, \c letter, \a alphanumeric.
LEGAL_PATTERNS = [
    r"PSL-\n\n\n\n-\n\n\n\n",       # matter id like PSL-2026-0418
    r"\$\n\n\n,\n\n\n",                 # currency amounts
    r"\$\n\n\n,\n\n\n.\n\n",
    r"Section \n",
    r"Article \n",
]


def write_user_words(words: list[str] | None = None) -> Path:
    target = settings.data_dir / "user_words.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("\n".join(words or LEGAL_TERMS), encoding="utf-8")
    return target


def write_user_patterns(patterns: list[str] | None = None) -> Path:
    target = settings.data_dir / "user_patterns.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("\n".join(patterns or LEGAL_PATTERNS), encoding="utf-8")
    return target


def tesseract_config_for_legal() -> str:
    """Return a Tesseract config string that biases toward legal vocabulary."""
    words_path = write_user_words()
    patterns_path = write_user_patterns()
    return (
        f'--user-words "{words_path}" '
        f'--user-patterns "{patterns_path}" '
        '-c preserve_interword_spaces=1'
    )
