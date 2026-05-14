"""Provider ABCs. Anything we depend on at a boundary lives behind one.

Swapping Anthropic for OpenAI, MiniLM for BGE, ChromaDB for Qdrant, or
Tesseract for AWS Textract means writing one new class and one line in
the registry. No call site changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def complete_vision(
        self,
        system: str,
        image_png: bytes,
        instruction: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0-1


class OCRProvider(ABC):
    @abstractmethod
    def ocr(self, image_png: bytes) -> OCRResult:
        ...


@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float
    ocr_confidence: float = 1.0


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, items: list[dict]) -> int:
        """items: [{chunk_id, doc_id, page, text, ocr_confidence, embedding}]"""
        ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[StoredChunk]:
        ...

    @abstractmethod
    def get_text(self, chunk_id: str) -> str | None:
        ...

    @abstractmethod
    def delete_doc(self, doc_id: str) -> int:
        """Remove all chunks belonging to `doc_id`. Returns number deleted."""
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
