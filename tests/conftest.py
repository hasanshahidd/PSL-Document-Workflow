"""Test fixtures: isolated SQLite per test, fake providers, env hygiene."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

# put project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Each test gets its own data dir + a fresh SQLite connection.

    We have to explicitly reset the module-level singletons in `config`,
    `storage.db`, and the provider registry because other modules already
    hold direct references to them — popping from sys.modules isn't enough.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # rebind config.settings to the new env
    import config
    config.settings = config.Settings()
    for d in (config.settings.raw_dir, config.settings.processed_dir,
              config.settings.chroma_dir, config.settings.samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    # nuke the cached SQLite connection so the next get_conn() opens the new db
    import storage.db as db_mod
    if db_mod._conn is not None:
        try:
            db_mod._conn.close()
        except Exception:
            pass
        db_mod._conn = None

    # reset provider registry so fake_llm fixture can re-bind cleanly
    from providers import registry
    registry.reset_all()

    yield


@pytest.fixture
def fake_llm():
    """Hook into the provider registry with a scriptable fake LLM."""
    from providers.registry import set_llm, reset_all
    from providers.base import LLMProvider, LLMResponse

    class FakeLLM(LLMProvider):
        def __init__(self):
            self.calls: list[dict] = []
            self.responses: list[str] = []

        def queue(self, *texts: str) -> None:
            self.responses.extend(texts)

        def _next(self) -> str:
            return self.responses.pop(0) if self.responses else "{}"

        def complete(self, system, messages, max_tokens=1024) -> LLMResponse:
            self.calls.append({"system": system, "messages": messages})
            return LLMResponse(text=self._next(), input_tokens=10, output_tokens=10,
                               model="fake", cached=False)

        def complete_vision(self, system, image_png, instruction, max_tokens=4096) -> LLMResponse:
            self.calls.append({"system": system, "vision": True})
            return LLMResponse(text=self._next(), input_tokens=10, output_tokens=10,
                               model="fake", cached=False)

    fake = FakeLLM()
    set_llm(fake)
    yield fake
    reset_all()
