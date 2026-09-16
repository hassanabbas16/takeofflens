"""Plan upload, status, page images and export."""

from __future__ import annotations

import csv
import io
import logging
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db import get_db
from app.jobs import process_plan
from app.models import OcrTokenRow, Page, Plan, Room
from app.pipeline.persist import create_plan
from app.schemas import PageOut, PlanOut, RoomOut, TokenOut
from app.storage import LocalStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plans", tags=["plans"])

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

# Columns in the CSV export, in order. area_m2 comes before width/length because it is the
# primary field on these plans - the others are usually null.
CSV_COLUMNS = [
    "plan_id", "filename", "page_number", "source", "name", "room_type",
    "area_m2", "width_m", "length_m", "confidence", "grounded",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "raw_text", "source_token_ids",
]


def _load_plan(db: Session, plan_id: UUID) -> Plan:
    plan = db.execute(
        select(Plan)
        .where(Plan.id == plan_id)
        .options(
            selectinload(Plan.pages).selectinload(Page.rooms),
            selectinload(Plan.pages).selectinload(Page.ocr_tokens),
        )
    ).scalar_one_or_none()
    if plan is None:
        raise HTTPException(status_code=404, detail=f"plan {plan_id} not found")
    return plan


@router.post("", status_code=202)
async def upload_plan(
    background: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Accept a plan and start processing it in the background.

    Returns 202 rather than 200: the work has been accepted, not completed. The client
    polls GET /plans/{id} for status.
    """
    settings = get_settings()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix or '(none)'}; allowed: pdf, png, jpg",
        )
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"unsupported content type {file.content_type}"
        )

    # Read in chunks and stop at the limit rather than loading the whole body first: an
    # oversized upload should cost us the limit, not the attacker's chosen size.
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {settings.max_upload_mb} MB limit",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    plan = create_plan(db, file.filename or f"upload{suffix}")
    db.commit()

    # The uploaded bytes outlive the request, so they go to a real file the background task
    # can still read after the response has been sent.
    upload_dir = Path(tempfile.gettempdir()) / "takeofflens_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_path = upload_dir / f"{plan.id}{suffix}"
    source_path.write_bytes(data)

    background.add_task(process_plan, plan.id, source_path)
    return JSONResponse(status_code=202, content={"id": str(plan.id), "status": plan.status})


@router.get("/{plan_id}", response_model=PlanOut)
def get_plan(
    plan_id: UUID,
    include_tokens: bool = Query(True, description="include OCR tokens for the overlay"),
    source: str | None = Query(None, description="filter rooms by extraction approach"),
    db: Session = Depends(get_db),
) -> PlanOut:
    plan = _load_plan(db, plan_id)
    return PlanOut(
        id=plan.id,
        filename=plan.filename,
        status=plan.status,
        error=plan.error,
        created_at=plan.created_at,
        pages=[
            PageOut(
                id=page.id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                tokens=(
                    [
                        TokenOut(
                            id=t.token_index,
                            text=t.text,
                            confidence=t.confidence,
                            bbox=list(t.bbox),
                        )
                        for t in page.ocr_tokens
                    ]
                    if include_tokens
                    else []
                ),
                rooms=[
                    _room_out(room)
                    for room in page.rooms
                    if source is None or room.source == source
                ],
            )
            for page in plan.pages
        ],
    )


def _room_out(room: Room) -> RoomOut:
    return RoomOut(
        id=room.id,
        name=room.name,
        room_type=room.room_type,
        width_m=room.width_m,
        length_m=room.length_m,
        area_m2=room.area_m2,
        source=room.source,
        raw_text=room.raw_text,
        bbox=room.bbox,
        confidence=room.confidence,
        grounded=room.grounded,
    )


@router.get("/{plan_id}/pages/{page_number}/image")
def get_page_image(
    plan_id: UUID, page_number: int, db: Session = Depends(get_db)
) -> Response:
    page = db.execute(
        select(Page).where(Page.plan_id == plan_id, Page.page_number == page_number)
    ).scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=404, detail=f"page {page_number} not found")

    storage = LocalStorage(get_settings().storage_dir)
    if not storage.exists(page.image_path):
        raise HTTPException(status_code=404, detail="page image missing from storage")
    return Response(
        content=storage.get(page.image_path),
        media_type="image/png",
        # Page images never change once written, so they can be cached hard.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{plan_id}/export")
def export_plan(
    plan_id: UUID,
    format: str = Query("csv", pattern="^(csv|json)$"),
    source: str | None = Query(None, description="filter by approach, e.g. rules or hybrid"),
    db: Session = Depends(get_db),
) -> Response:
    plan = _load_plan(db, plan_id)
    rows = [
        (page, room)
        for page in plan.pages
        for room in page.rooms
        if source is None or room.source == source
    ]

    if format == "json":
        payload = {
            "plan_id": str(plan.id),
            "filename": plan.filename,
            "status": plan.status,
            "source_filter": source,
            "rooms": [
                {
                    "page_number": page.page_number,
                    **_room_out(room).model_dump(mode="json"),
                }
                for page, room in rows
            ],
        }
        return JSONResponse(content=payload)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for page, room in rows:
        bbox = room.bbox or [None, None, None, None]
        writer.writerow({
            "plan_id": str(plan.id),
            "filename": plan.filename,
            "page_number": page.page_number,
            "source": room.source,
            "name": room.name,
            "room_type": room.room_type,
            "area_m2": room.area_m2,
            "width_m": room.width_m,
            "length_m": room.length_m,
            "confidence": room.confidence,
            "grounded": room.grounded,
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "raw_text": room.raw_text,
            "source_token_ids": room.source_token_ids,
        })
    buffer.seek(0)
    stem = Path(plan.filename).stem or "plan"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_takeoff.csv"'},
    )


@router.get("/{plan_id}/tokens")
def get_tokens(plan_id: UUID, db: Session = Depends(get_db)) -> list[TokenOut]:
    """Every OCR token across the plan, for the overlay."""
    tokens = db.execute(
        select(OcrTokenRow)
        .join(Page, Page.id == OcrTokenRow.page_id)
        .where(Page.plan_id == plan_id)
        .order_by(Page.page_number, OcrTokenRow.token_index)
    ).scalars().all()
    return [
        TokenOut(id=t.token_index, text=t.text, confidence=t.confidence, bbox=list(t.bbox))
        for t in tokens
    ]
