"""Re-score the cached Tier 3 rows from the raw batch output, without paying again.

The per-(plan, approach) cache rows hold scores, not model output, so a change to grounding
or scoring leaves them stale - and the Tier 3 report is built from them. This replays
scoring against the raw batch bodies saved by ``eval/retrieve_batch.py`` and rewrites the
rows in place, preserving the token counts and cost that were actually billed.

Only rows whose raw body is available are touched; `rules` rows are recomputed from OCR.

    python eval/rescore_from_raw.py
    python eval/tier3_cost_probe.py --sample 50 --batch --out /eval/results/tier3_sample50.md

No API calls. Free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, "/eval")

from app.pipeline.extract import ExtractionOutcome, ground_hybrid, ground_ocr_llm, ground_vlm
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef
from app.schemas import PlanExtraction
from tier3_cost_probe import (
    CACHE_DIR,
    OCR_CACHE_DIR,
    PRINTED_AREAS_PATH,
    _SOURCE_FOR,
    plan_dir,
    reference,
    score,
)

RAW_DIR = Path("/eval/cache/batch_raw")
SLUG_TO_APPROACH = {"ocr_llm": "ocr+llm", "vlm": "vlm", "hybrid": "hybrid"}
GROUND = {"ocr+llm": ground_ocr_llm, "vlm": ground_vlm, "hybrid": ground_hybrid}


def load_tokens(plan_id: str) -> list[OcrToken] | None:
    path = OCR_CACHE_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tokens" not in data:
        return None
    return [
        OcrToken(
            text=t["text"], confidence=t["confidence"], bbox=tuple(t["bbox"]),
            angle=t.get("angle", 0),
        )
        for t in data["tokens"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    args = parser.parse_args()

    printed = json.loads(PRINTED_AREAS_PATH.read_text(encoding="utf-8"))["plans"]
    changed = 0
    seen = 0

    for raw_path in sorted(RAW_DIR.glob("*.json")):
        entry = json.loads(raw_path.read_text(encoding="utf-8"))
        if "text" not in entry:
            continue
        head = raw_path.stem.rsplit("-", 1)[0]
        plan_id, _, slug = head.partition("_")
        approach = SLUG_TO_APPROACH.get(slug, slug)
        if approach not in GROUND:
            continue

        tokens = load_tokens(plan_id)
        if tokens is None:
            continue
        refs = TokenRef.from_tokens(tokens)
        try:
            extraction = PlanExtraction.model_validate(json.loads(entry["text"]))
        except Exception:  # noqa: BLE001 - leave an unparseable row exactly as it was
            continue

        grounded = GROUND[approach](extraction, refs)
        outcome = ExtractionOutcome(_SOURCE_FOR[approach], grounded, None, refs)
        directory = plan_dir(args.dataset, f"/high_quality_architectural/{plan_id}/")
        labels, areas = reference(directory / "model.svg")
        matched_labels, matched_areas = score(outcome, labels, areas)
        n_printed = printed.get(plan_id, {}).get("printed_area_count", 0)

        cache = CACHE_DIR / approach.replace("+", "_") / f"{plan_id}.json"
        if not cache.exists():
            continue
        row = json.loads(cache.read_text(encoding="utf-8"))
        seen += 1
        old = (row.get("labels"), row.get("areas"), row.get("hallucinations"))
        row.update({
            "rooms": len(outcome.rooms),
            "labels": matched_labels,
            "ref_labels": len(labels),
            "areas": matched_areas,
            "printed_areas": n_printed,
            "areas_on_printing_plans": matched_areas if n_printed else 0,
            "spurious_areas": 0 if n_printed else matched_areas,
            "hallucinations": outcome.hallucinations,
            "rescored": True,
        })
        new = (row["labels"], row["areas"], row["hallucinations"])
        if old != new:
            changed += 1
            print(f"  {plan_id:>6} {approach:<8} labels {old[0]}->{new[0]}  "
                  f"areas {old[1]}->{new[1]}  halluc {old[2]}->{new[2]}")
        cache.write_text(json.dumps(row, indent=2), encoding="utf-8")

    print(f"\nrescored {seen} rows, {changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
