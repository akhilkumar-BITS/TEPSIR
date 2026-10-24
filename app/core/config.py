"""Application settings, loaded from the environment (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    project_name: str = "TEPSIR"
    api_v1_prefix: str = "/api/v1"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://localhost:4173",
    ]

    # Security
    secret_key: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    password_min_length: int = 8

    # Database
    postgres_dsn: str = "postgresql+asyncpg://tepsir:tepsir@localhost:5432/tepsir"
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Search orchestration / cache (optional; search still works without Redis)
    search_cache_enabled: bool = True
    redis_url: str | None = "redis://localhost:6379/0"
    embedding_cache_ttl_seconds: int = 3600

    # Object storage
    storage_backend: Literal["local", "minio"] = "local"
    local_storage_dir: Path = Path("var/storage")
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "tepsir"
    minio_secret_key: str = "tepsir-secret"
    minio_secure: bool = False
    minio_bucket: str = "tepsir-documents"

    # Retrieval
    embedding_provider: Literal["dev", "sentence-transformers"] = "dev"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_upload_bytes: int = 25 * 1024 * 1024
    search_top_k_default: int = 5
    search_top_k_max: int = 20
    max_search_query_length: int = 500

    # Bootstrap admin (created on startup when both are set)
    seed_admin_email: str | None = None
    seed_admin_password: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
