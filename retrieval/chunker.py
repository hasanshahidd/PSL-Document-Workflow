"""Paragraph-aware sliding-window chunker that preserves page provenance.

We chunk per page so every chunk has an unambiguous page number for citation.
Within a page we split on blank lines and pack paragraphs up to a soft token
budget (approximated by word count). Adjacent chunks overlap by a small window
so a claim split across chunk boundaries is still retrievable.
"""
from __future__ import annotations
import re
from schemas import Chunk, PageContent, ProcessedDocument


TARGET_WORDS = 280  # ~400 tokens at typical legal-text ratio
OVERLAP_WORDS = 60


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    word_count = 0
    for para in paragraphs:
        n = len(para.split())
        if buf and word_count + n > TARGET_WORDS:
            chunks.append("\n\n".join(buf))
            # carry overlap from tail of previous chunk
            tail_words = " ".join(buf).split()[-OVERLAP_WORDS:]
            buf = [" ".join(tail_words)] if tail_words else []
            word_count = len(tail_words)
        buf.append(para)
        word_count += n
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def chunk_page(doc_id: str, page: PageContent) -> list[Chunk]:
    paragraphs = _split_paragraphs(page.text)
    pieces = _pack(paragraphs) if paragraphs else []
    return [
        Chunk(
            chunk_id=f"{doc_id}:p{page.page}:c{i}",
            doc_id=doc_id,
            page=page.page,
            text=piece,
            ocr_confidence=page.ocr_confidence,
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_document(doc: ProcessedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(chunk_page(doc.doc_id, page))
    return chunks
