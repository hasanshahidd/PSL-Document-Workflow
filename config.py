from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider -------------------------------------------------------
    # "auto" picks based on which API key is set (OpenAI wins if both are set).
    llm_provider: Literal["auto", "openai", "anthropic"] = "auto"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # --- OCR + paths --------------------------------------------------------
    tesseract_cmd: str | None = None
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ocr_confidence_threshold: float = 0.60
    demo_mode: bool = False

    # --- helpers ------------------------------------------------------------
    @property
    def active_llm_provider(self) -> Literal["openai", "anthropic"]:
        """Resolve `auto` to a concrete provider name based on which key is set."""
        if self.llm_provider == "auto":
            if self.openai_api_key:
                return "openai"
            if self.anthropic_api_key:
                return "anthropic"
            return "openai"  # default; will raise at use time if no key
        return self.llm_provider

    @property
    def active_llm_model(self) -> str:
        return self.openai_model if self.active_llm_provider == "openai" else self.anthropic_model

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"


settings = Settings()
for d in (settings.raw_dir, settings.processed_dir, settings.chroma_dir, settings.samples_dir):
    d.mkdir(parents=True, exist_ok=True)
