"""Background processing for an uploaded plan.

Runs as a FastAPI BackgroundTask: the upload returns immediately with a plan id and the
work happens after the response. The interface is deliberately a single function taking a
plan id and owning its own database session, so moving to a real queue later means calling
the same function from a worker instead of from BackgroundTasks - nothing else changes.
That migration note is in the README.

Which approaches run is configuration, not a hardcoded list, and the default is ``rules``
only. An uploaded plan should not silently spend money on model calls; the paid approaches
are opted into with ``PIPELINE_APPROACHES``.

Nothing here raises. A failure sets ``status="failed"`` with a message, because a job that
crashes leaves a plan stuck in ``processing`` forever with no explanation.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from app.config import get_settings
from app.db import SessionLocal
from app.llm import LlmClient, LlmError
from app.models import Plan
from app.pipeline.extract import (
    ExtractionOutcome,
    extract_hybrid,
    extract_ocr_llm,
    extract_rules,
    extract_vlm,
)
from app.pipeline.ingest import IngestError, ingest
from app.pipeline.ocr import OcrToken, run_ocr
from app.pipeline.persist import add_page, save_llm_call, save_rooms, save_tokens, set_status
from app.pipeline.preprocess import PreprocessConfig, preprocess
from app.storage import LocalStorage

logger = logging.getLogger(__name__)


def page_image_key(plan_id: UUID, page_number: int) -> str:
    return f"plans/{plan_id}/page_{page_number:03d}.png"


def _preprocess_config() -> PreprocessConfig:
    settings = get_settings()
    return PreprocessConfig(
        grayscale=settings.preprocess_grayscale,
        denoise=settings.preprocess_denoise,
        adaptive_threshold=settings.preprocess_adaptive_threshold,
        deskew=settings.preprocess_deskew,
        upscale=settings.preprocess_upscale,
        upscale_min_short_side=settings.preprocess_upscale_min_short_side,
    )


def process_plan(plan_id: UUID, source_path: Path) -> None:
    """Ingest, OCR and extract one uploaded plan. Never raises."""
    settings = get_settings()
    storage = LocalStorage(settings.storage_dir)
    db = SessionLocal()

    try:
        plan = db.get(Plan, plan_id)
        if plan is None:
            logger.error("plan %s vanished before processing", plan_id)
            return
        set_status(db, plan, "processing")
        db.commit()

        try:
            pages = ingest(source_path, dpi=settings.ingest_dpi)
        except IngestError as exc:
            set_status(db, plan, "failed", str(exc))
            db.commit()
            return

        config = _preprocess_config()
        approaches = settings.pipeline_approach_list
        client: LlmClient | None = None
        if any(a != "rules" for a in approaches):
            try:
                client = LlmClient()
            except LlmError as exc:
                # Not fatal: the free approach still produces a useful result, and failing
                # the whole job over a missing key would be worse than a partial one.
                logger.warning("model approaches disabled: %s", exc)
                approaches = [a for a in approaches if a == "rules"]

        for page in pages:
            ok, buffer = cv2.imencode(".png", page.image)
            if not ok:
                set_status(db, plan, "failed", f"could not encode page {page.page_number}")
                db.commit()
                return
            key = page_image_key(plan_id, page.page_number)
            storage.put(key, buffer.tobytes())

            row = add_page(db, plan, page.page_number, key, page.width, page.height)

            processed = preprocess(page.image, config)
            tokens = run_ocr(
                processed,
                lang=settings.ocr_lang,
                orientations=settings.ocr_orientation_list,
                det_limit_side_len=settings.ocr_det_limit_side_len,
                det_limit_type=settings.ocr_det_limit_type,
                min_confidence=settings.ocr_min_confidence,
                dedup_iou=settings.ocr_dedup_iou,
                dedup_aspect_rule=settings.ocr_dedup_aspect_rule,
            )
            save_tokens(db, row, tokens)

            for approach in approaches:
                outcome = _run_approach(approach, tokens, page.image, client)
                if outcome is None:
                    continue
                if approach != "rules":
                    save_llm_call(db, row, approach, outcome)
                if outcome.ok:
                    save_rooms(db, row, outcome)
                else:
                    logger.warning(
                        "plan %s page %s approach %s failed: %s",
                        plan_id, page.page_number, approach, outcome.error,
                    )
            db.commit()

        set_status(db, plan, "done")
        db.commit()

    except Exception as exc:
        logger.exception("processing failed for plan %s", plan_id)
        db.rollback()
        try:
            plan = db.get(Plan, plan_id)
            if plan is not None:
                detail = f"{type(exc).__name__}: {exc}"
                set_status(db, plan, "failed", detail)
                db.commit()
        except Exception:  # noqa: BLE001 - last resort; nowhere left to report to
            logger.error("could not record failure for %s\n%s", plan_id, traceback.format_exc())
    finally:
        db.close()


def _run_approach(
    approach: str,
    tokens: list[OcrToken],
    image: np.ndarray,
    client: LlmClient | None,
) -> ExtractionOutcome | None:
    if approach == "rules":
        return extract_rules(tokens)
    if client is None:
        return None
    if approach == "ocr+llm":
        return extract_ocr_llm(tokens, client)
    if approach == "vlm":
        return extract_vlm(image, client, tokens=tokens)
    if approach == "hybrid":
        return extract_hybrid(image, tokens, client)
    logger.warning("unknown approach %r, skipping", approach)
    return None
