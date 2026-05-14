from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, computed_field


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class OcrDiagnostics(BaseModel):
    tesseract_confidence: float = 0.0
    sanity_score: float = 0.0
    rotation_corrected: int = 0
    chosen_variant: str | None = None
    chosen_psm: int | None = None
    preprocess_stages: list[str] = []
    variants: list[dict] = []
    needs_vision_fallback: bool = False


class PageContent(BaseModel):
    page: int
    text: str
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["digital", "tesseract", "vision"]
    low_conf_regions: list[BBox] = []
    ocr_diagnostics: OcrDiagnostics | None = None


class StructuredFields(BaseModel):
    doc_type: str | None = None
    parties: list[str] = []
    dates: list[str] = []
    matter_id: str | None = None
    extra: dict = {}


class ProcessedDocument(BaseModel):
    doc_id: str
    filename: str
    pages: list[PageContent]
    fields: StructuredFields
    full_text: str

    @property
    def mean_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.ocr_confidence for p in self.pages) / len(self.pages)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    text: str
    ocr_confidence: float = 1.0


class EvidenceChunk(Chunk):
    score: float = 0.0


class Citation(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    quote: str | None = None  # exact supporting sentence pulled from the chunk


class DraftSentence(BaseModel):
    idx: int
    text: str
    citations: list[Citation] = []
    supported: bool = True


class Draft(BaseModel):
    draft_id: str
    draft_type: str
    doc_ids: list[str]
    sentences: list[DraftSentence]
    uncertain_sections: list[str] = []

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sentences)

    @computed_field
    @property
    def grounding_rate(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(1 for s in self.sentences if s.supported) / len(self.sentences)
