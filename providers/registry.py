"""Dependency-injection container.

Single source of truth for the four providers. Every other module gets its
provider through `get_llm()`, `get_embedder()`, etc. Tests inject doubles by
calling `set_llm(FakeLLM())` before the system-under-test runs.
"""
from __future__ import annotations
from providers.base import EmbeddingProvider, LLMProvider, OCRProvider, VectorStore


_llm: LLMProvider | None = None
_embedder: EmbeddingProvider | None = None
_ocr: OCRProvider | None = None
_vector_store: VectorStore | None = None


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        from config import settings
        if settings.active_llm_provider == "openai":
            from providers.openai_llm import OpenAILLM
            _llm = OpenAILLM()
        else:
            from providers.anthropic_llm import AnthropicLLM
            _llm = AnthropicLLM()
    return _llm


def set_llm(provider: LLMProvider) -> None:
    global _llm
    _llm = provider


def get_embedder() -> EmbeddingProvider:
    global _embedder
    if _embedder is None:
        from providers.st_embedder import STEmbedder
        _embedder = STEmbedder()
    return _embedder


def set_embedder(provider: EmbeddingProvider) -> None:
    global _embedder
    _embedder = provider


def get_ocr() -> OCRProvider:
    global _ocr
    if _ocr is None:
        from providers.tesseract_ocr import TesseractOCR
        _ocr = TesseractOCR()
    return _ocr


def set_ocr(provider: OCRProvider) -> None:
    global _ocr
    _ocr = provider


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from providers.chroma_store import ChromaVectorStore
        _vector_store = ChromaVectorStore()
    return _vector_store


def set_vector_store(provider: VectorStore) -> None:
    global _vector_store
    _vector_store = provider


def reset_all() -> None:
    """Test helper. Clears the container so the next get_* re-bootstraps."""
    global _llm, _embedder, _ocr, _vector_store
    _llm = _embedder = _ocr = _vector_store = None
