from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-interview-api"
    app_env: str = "development"
    log_level: str = "INFO"

    # Gemini
    gemini_api_key: SecretStr
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, gt=0)

    # Metadata database
    database_url: str = "sqlite+aiosqlite:///./app.db"

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_history_max_messages: int = Field(default=20, gt=0)
    chat_ttl_seconds: int = Field(default=3600, gt=0)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # Ingestion / retrieval
    max_upload_mb: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_score_threshold: float | None = None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()