from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production RAG Assistant"
    environment: str = "development"
    log_level: str = "INFO"

    docs_path: Path = Path("docs.json")
    database_path: Path = Path("data/rag.db")

    embedding_provider: Literal["openai", "gemini", "mistral", "local"] = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 384
    embedding_timeout_seconds: float = 20.0

    llm_provider: Literal["openai", "gemini", "anthropic", "mistral", "local"] = "local"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.2, ge=0, le=0.3)
    llm_timeout_seconds: float = 30.0

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 60
    top_k: int = 3
    similarity_threshold: float = 0.28
    conversation_pairs: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
