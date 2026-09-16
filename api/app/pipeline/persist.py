"""Persist pipeline results to the database.

Kept separate from the extraction modules so the approaches stay runnable without a database
(the CLI and the eval harness both do exactly that).
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models import LlmCall, OcrTokenRow, Page, Plan, Room
from app.pipeline.extract import ExtractionOutcome
from app.pipeline.grounding import grounded_room_ids, room_bbox_from_tokens
from app.pipeline.ocr import OcrToken
from app.pipeline.room_types import primary_type


def create_plan(db: Session, filename: str) -> Plan:
    plan = Plan(filename=filename, status="pending")
    db.add(plan)
    db.flush()
    return plan


def set_status(db: Session, plan: Plan, status: str, error: str | None = None) -> None:
    plan.status = status
    plan.error = error
    db.add(plan)
    db.flush()


def add_page(
    db: Session, plan: Plan, page_number: int, image_key: str, width: int, height: int
) -> Page:
    page = Page(
        plan_id=plan.id,
        page_number=page_number,
        image_path=image_key,
        width=width,
        height=height,
    )
    db.add(page)
    db.flush()
    return page


def save_tokens(db: Session, page: Page, tokens: list[OcrToken]) -> None:
    """Store tokens with the same index the LLM was asked to cite."""
    for index, token in enumerate(tokens):
        x1, y1, x2, y2 = token.bbox
        db.add(
            OcrTokenRow(
                page_id=page.id,
                token_index=index,
                text=token.text,
                confidence=token.confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                angle=token.angle,
            )
        )
    db.flush()


def save_llm_call(
    db: Session, page: Page | None, purpose: str, outcome: ExtractionOutcome
) -> LlmCall:
    usage = outcome.usage
    call = LlmCall(
        page_id=page.id if page is not None else None,
        # A retry costs real tokens, so record that it happened.
        purpose=purpose if usage.attempts <= 1 else f"{purpose}:retry",
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        latency_ms=usage.latency_ms,
        ok=usage.ok,
        error=usage.error,
    )
    db.add(call)
    db.flush()
    return call


def save_rooms(db: Session, page: Page, outcome: ExtractionOutcome) -> list[Room]:
    """Persist extracted rooms, including ungrounded ones flagged as such."""
    if outcome.grounded is None:
        return []

    ungrounded = {id(room) for room in outcome.grounded.dropped}
    saved: list[Room] = []

    for room in outcome.grounded.rooms:
        bbox = room_bbox_from_tokens(room, outcome.refs)
        # Trust the lexicon over the model for the type when the label is one we know.
        mapped = primary_type(room.label_raw)
        room_type = mapped if mapped != "unknown" else room.room_type.value
        entity = Room(
            page_id=page.id,
            name=room.label_raw or None,
            room_type=room_type,
            width_m=room.width_m,
            length_m=room.length_m,
            area_m2=room.area_m2,
            source=outcome.source.value,
            raw_text=room.label_raw,
            x1=bbox[0] if bbox else None,
            y1=bbox[1] if bbox else None,
            x2=bbox[2] if bbox else None,
            y2=bbox[3] if bbox else None,
            confidence=room.confidence,
            source_token_ids=grounded_room_ids(room),
            grounded=id(room) not in ungrounded,
        )
        db.add(entity)
        saved.append(entity)

    db.flush()
    return saved


def get_plan(db: Session, plan_id: uuid.UUID) -> Plan | None:
    return db.get(Plan, plan_id)
