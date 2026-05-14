"""Embedding shim — kept for backwards compatibility. Delegates to the provider."""
from __future__ import annotations
from providers.registry import get_embedder


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed(texts)
