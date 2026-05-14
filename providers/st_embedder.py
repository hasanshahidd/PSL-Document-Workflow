"""sentence-transformers embedding provider, traced."""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from config import settings
from obs.tracer import span
from providers.base import EmbeddingProvider


class STEmbedder(EmbeddingProvider):
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with span("embed", model=self._model_name, n=len(texts)):
            vecs = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()
