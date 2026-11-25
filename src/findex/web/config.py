"""Environment-driven settings for findex web service."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Web service configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FINDEX_",
        extra="ignore",
    )

    index_path: Path = Field(
        default=Path("index.json"),
        validation_alias="INDEX_PATH",
        description="Path to serialized JSON index loaded on startup.",
    )
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, ge=1, le=65535, validation_alias="PORT")
    workers: int = Field(default=1, ge=1, validation_alias="WORKERS")
    default_top_k: int = Field(default=10, ge=1, le=100)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance read from environment."""
    return Settings()