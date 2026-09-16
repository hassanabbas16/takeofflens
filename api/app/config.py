"""Application settings. Every environment-specific value lives here, never inline."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The project .env, mounted read-only into the container. API keys are read from THIS FILE
# ONLY - never from the process environment. A shell that happens to export ANTHROPIC_API_KEY
# (a developer machine, a CI runner, an agent session) would otherwise silently supply a
# different key than the one the project is configured with, and the first cost-gate run in
# this project did exactly that.
ENV_FILE = Path(os.environ.get("TAKEOFFLENS_ENV_FILE", "/app/.env"))


def read_env_file_value(key: str, path: Path = ENV_FILE) -> str:
    """Read one value from the .env file, ignoring the process environment entirely.

    Returns "" when the file or key is absent, so callers fail with a clear message rather
    than falling back to an ambient credential.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != key:
            continue
        value = value.strip()
        # Strip optional surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://takeofflens:change_me_locally@db:5432/takeofflens"

    # Anthropic. Model names are env-driven so they are never hardcoded in pipeline logic.
    # NOTE: the API key is deliberately NOT a settings field - see api_key() below.
    anthropic_text_model: str = ""
    anthropic_vision_model: str = ""
    anthropic_timeout_seconds: float = 120.0
    anthropic_max_retries: int = 1  # one retry on schema failure, per the spec
    # max_tokens is required by the Messages API. A plan has at most a few dozen rooms;
    # 8000 leaves generous headroom, and hitting the cap is treated as a failure because a
    # truncated room list looks like a complete one.
    anthropic_max_tokens: int = 8000
    # Temperature 0 for determinism. Sonnet 5 and Opus 5 removed sampling parameters and
    # reject this; that is detected at runtime and the parameter dropped - see llm.py.
    anthropic_temperature: float | None = 0.0
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
    def anthropic_api_key(self) -> str:
        """The API key, read from the project .env file and nowhere else.

        Deliberately not a pydantic-settings field: those read the process environment, and
        an ambient ANTHROPIC_API_KEY must never be able to pay for this project's calls.
        """
        return read_env_file_value("ANTHROPIC_API_KEY")

    @property
    def api_key_source(self) -> str:
        """Where the key came from, for reporting. Never returns the key itself."""
        key = self.anthropic_api_key
        if not key:
            return f"absent (looked in {ENV_FILE})"
        if key.startswith("sk-ant-replace"):
            return f"placeholder in {ENV_FILE}"
        return f"{ENV_FILE} (prefix {key[:14]}..., {len(key)} chars)"

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
