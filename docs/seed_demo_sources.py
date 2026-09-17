"""Seed a plan's `vlm` and `hybrid` rooms into the database from the eval cache.

**Demo tooling. Not part of the pipeline, and it makes no API calls.**

The interactive upload path runs `PIPELINE_APPROACHES`, which defaults to `rules` alone, so
an uploaded plan has one source and the viewer's approach toggle has nothing to toggle.
Running `vlm` and `hybrid` live would cost money.

This replays the already-paid-for batch output in `eval/cache/batch_raw/` for one plan, and
writes the resulting rooms against an uploaded plan's page so the viewer can show the same
approach comparison the evaluation reports. The rooms are the real extraction output, passed
through the real grounding and persistence code - nothing is fabricated for the camera.

    python docs/seed_demo_sources.py --plan-id <uuid> --source-plan 1191
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import UUID

sys.path.insert(0, "/app")
sys.path.insert(0, "/eval")

from sqlalchemy import select

from app.db import SessionLocal
from app.models import OcrTokenRow, Page, Room
from app.pipeline.extract import ExtractionOutcome, ground_hybrid, ground_vlm
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef
from app.schemas import PlanExtraction, Source

RAW_DIR = Path("/eval/cache/batch_raw")
GROUND = {"vlm": ground_vlm, "hybrid": ground_hybrid}
SOURCE = {"vlm": Source.VLM, "hybrid": Source.HYBRID}


def cached_extraction(source_plan: str, approach: str) -> PlanExtraction | None:
    for path in RAW_DIR.glob(f"{source_plan}_{approach}-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if "text" not in entry:
            continue
        try:
            return PlanExtraction.model_validate(json.loads(entry["text"]))
        except Exception:  # noqa: BLE001 - a bad body means no seed, not a crash
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-id", required=True, help="uploaded plan uuid")
    parser.add_argument("--source-plan", default="1191", help="dataset plan id in the cache")
    parser.add_argument("--wait-seconds", type=float, default=120.0,
                        help="how long to wait for ingest to create the page row")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # The recorder calls this straight after upload, so ingest may not have written the
        # page row yet. Wait for it rather than failing the recording.
        deadline = time.monotonic() + args.wait_seconds
        page = None
        while page is None and time.monotonic() < deadline:
            page = db.scalars(
                select(Page)
                .where(Page.plan_id == UUID(args.plan_id))
                .order_by(Page.page_number)
            ).first()
            if page is None:
                db.rollback()
                time.sleep(1.0)
        if page is None:
            print(f"no page for plan {args.plan_id} after {args.wait_seconds:g}s",
                  file=sys.stderr)
            return 1

        # OCR tokens are written with the page, but wait for a non-empty set: the cited
        # token ids are what the seeded rooms' boxes are derived from.
        rows = []
        while not rows and time.monotonic() < deadline:
            rows = db.scalars(
                select(OcrTokenRow)
                .where(OcrTokenRow.page_id == page.id)
                .order_by(OcrTokenRow.token_index)
            ).all()
            if not rows:
                db.rollback()
                time.sleep(1.0)

        tokens = [
            OcrToken(text=r.text, confidence=r.confidence, bbox=(r.x1, r.y1, r.x2, r.y2),
                     angle=0)
            for r in rows
        ]
        refs = TokenRef.from_tokens(tokens)

        added = 0
        for approach in ("vlm", "hybrid"):
            extraction = cached_extraction(args.source_plan, approach)
            if extraction is None:
                print(f"no cached {approach} result for plan {args.source_plan}")
                continue
            grounded = GROUND[approach](extraction, refs)
            outcome = ExtractionOutcome(SOURCE[approach], grounded, None, refs)
            for room in outcome.rooms:
                ids = getattr(room, "source_token_ids", None) or []
                boxes = [refs[i].bbox for i in ids if 0 <= i < len(refs)]
                bbox = None
                if boxes:
                    bbox = (
                        min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes),
                    )
                db.add(Room(
                    page_id=page.id,
                    name=room.label_raw,
                    room_type=getattr(room.room_type, "value", room.room_type) or "unknown",
                    width_m=room.width_m,
                    length_m=room.length_m,
                    area_m2=room.area_m2,
                    source=SOURCE[approach].value,
                    raw_text=room.label_raw,
                    x1=bbox[0] if bbox else None,
                    y1=bbox[1] if bbox else None,
                    x2=bbox[2] if bbox else None,
                    y2=bbox[3] if bbox else None,
                    confidence=room.confidence,
                    source_token_ids=",".join(str(i) for i in ids) or None,
                    grounded=room not in grounded.dropped,
                ))
                added += 1
            print(f"  {approach}: {len(outcome.rooms)} rooms from cache")
        db.commit()
    finally:
        db.close()

    print(f"seeded {added} rooms for plan {args.plan_id} (no API calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
