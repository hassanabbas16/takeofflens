"""Application settings. Every environment-specific value lives here, never inline."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://takeofflens:change_me_locally@db:5432/takeofflens"

    # OpenAI. Model names are env-driven so they are never hardcoded in pipeline logic.
    openai_api_key: str = ""
    openai_text_model: str = ""
    openai_vision_model: str = ""

    # Local storage root, behind the Storage interface so S3 can replace it later.
    storage_dir: Path = Path("/storage")

    # CubiCasa5K, mounted read-only. Never written to, never copied into the repo.
    dataset_dir: Path = Path("/data/cubicasa5k")
    dataset_coco_dir: Path = Path("/data/cubicasa5k_coco")

    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
