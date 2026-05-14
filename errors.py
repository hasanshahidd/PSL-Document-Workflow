"""Project exception hierarchy.

Every module raises a subclass of PSLError so the API layer can map exceptions
to HTTP status codes in one place. Internal callers can catch the narrow
subclass they care about without sprinkling try/except around every provider.
"""
from __future__ import annotations


class PSLError(Exception):
    """Base class for all expected, in-domain errors."""


class IngestionError(PSLError):
    """Raised when a document cannot be processed."""


class UnsupportedFormatError(IngestionError):
    pass


class OCRError(IngestionError):
    pass


class RetrievalError(PSLError):
    pass


class IndexError_(RetrievalError):
    """Renamed to avoid clashing with the builtin IndexError."""


class DraftingError(PSLError):
    pass


class ProviderError(PSLError):
    """Anything wrong at the provider boundary (LLM, embedder, vector store)."""


class LLMError(ProviderError):
    pass


class LLMRateLimited(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


class ConfigError(PSLError):
    pass
