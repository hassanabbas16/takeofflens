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
    openai_timeout_seconds: float = 120.0
    openai_max_retries: int = 1  # one retry on schema failure, per the spec
    # Temperature 0 for determinism. Models that reject the parameter are detected at
    # runtime and the parameter is dropped for them - see llm.py.
    openai_temperature: float = 0.0
    # Longest image edge sent to the vision models, to control cost.
    vlm_max_image_px: int = 1536
    vlm_jpeg_quality: int = 85
    # Pricing table used to turn logged tokens into a cost figure.
    pricing_path: Path = Path("/eval/pricing.yaml")

    # Local storage root, behind the Storage interface so S3 can replace it later.
    storage_dir: Path = Path("/storage")

    # CubiCasa5K, mounted read-only. Never written to, never copied into the repo.
    dataset_dir: Path = Path("/data/cubicasa5k")
    dataset_coco_dir: Path = Path("/data/cubicasa5k_coco")

    # ---- Ingest ----
    # PDF render DPI. 300 is the spec default; lower it for cheap smoke runs.
    ingest_dpi: int = 300

    # ---- Preprocess ----
    # Every step is individually toggleable so eval can ablate them (Phase 6).
    # Defaults are set from a measured sweep on a CubiCasa architectural plan; see
    # docs in preprocess.py for why adaptive threshold defaults off.
    preprocess_grayscale: bool = False
    preprocess_denoise: bool = False
    preprocess_adaptive_threshold: bool = False
    preprocess_deskew: bool = False
    preprocess_upscale: bool = False
    preprocess_upscale_min_short_side: int = 2000

    # ---- OCR ----
    ocr_lang: str = "en"
    # Orientations (degrees) each page is OCR'd at, then merged. Floor plan labels are
    # frequently set at 90 degrees, and a single pass misses most of them.
    ocr_orientations: str = "0,90,180,270"
    # Detection input cap. Counter-intuitively the default 1536 beats larger values on
    # these drawings; see ocr.py.
    ocr_det_limit_side_len: int = 1536
    ocr_det_limit_type: str = "max"
    ocr_min_confidence: float = 0.5
    # IoU above which two boxes from different orientation passes are the same token.
    ocr_dedup_iou: float = 0.5
    # How box aspect ratio constrains which orientation pass a reading may come from.
    # "tall_only" (default), "both" or "off" - see ocr.orientation_is_plausible.
    ocr_dedup_aspect_rule: str = "tall_only"

    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"

    @property
    def ocr_orientation_list(self) -> list[int]:
        return [int(a) for a in self.ocr_orientations.split(",") if a.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
