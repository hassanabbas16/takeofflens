"""Tier 4 step 3: area accuracy for all four approaches against the hand-verified gold set.

Runs entirely from cache - the gold JSON written by ``label_helper.py``, the OCR cache, and
the raw batch bodies saved by ``retrieve_batch.py``. **No API calls, no cost.** That is why
the gold plans were drawn from the 50-plan Tier 3 sample: their results are already paid for.

Reports, per approach, over the gold plans only:

precision
    of the areas an approach reported, how many are actually printed on the drawing.
    This is the number that catches fabrication.

recall
    of the areas actually printed, how many the approach found.

The two are reported separately and never averaged into one figure, because they fail in
opposite directions: `rules` is conservative and `vlm` is liberal, and a single score would
hide that. Matching is greedy within a relative tolerance and each gold area can be claimed
once, so reporting the same area five times cannot inflate recall.

Plans whose gold file is incomplete are skipped and named, so a partial labelling session
produces a partial - but honest - table rather than a wrong one.

    python eval/tier4_area_accuracy.py
    python eval/tier4_area_accuracy.py --tolerance 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.extract import (
    ExtractionOutcome,
    extract_rules,
    ground_hybrid,
    ground_ocr_llm,
    ground_vlm,
)
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef
from app.schemas import PlanExtraction, Source

GOLD_DIR = Path("/eval/ground_truth/gold")
OCR_CACHE_DIR = Path("/eval/cache/ocr")
RAW_DIR = Path("/eval/cache/batch_raw")
GOLD_LIST = Path("/eval/data/gold_15.txt")

APPROACHES = ["rules", "ocr+llm", "vlm", "hybrid"]
SLUG = {"ocr+llm": "ocr_llm", "vlm": "vlm", "hybrid": "hybrid"}
GROUND = {"ocr+llm": ground_ocr_llm, "vlm": ground_vlm, "hybrid": ground_hybrid}
SOURCE = {"ocr+llm": Source.OCR_LLM, "vlm": Source.VLM, "hybrid": Source.HYBRID}


def read_gold_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def load_tokens(plan_id: str) -> list[OcrToken] | None:
    path = OCR_CACHE_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tokens" not in data:
        return None
    return [
        OcrToken(text=t["text"], confidence=t["confidence"],
                 bbox=tuple(t["bbox"]), angle=t.get("angle", 0))
        for t in data["tokens"]
    ]


def gold_areas(plan_id: str) -> list[float] | None:
    """The areas the drawing prints, per the human. None if not finished."""
    path = GOLD_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        return None
    areas = [
        r["area_m2"] for r in payload["rooms"]
        if r.get("decided") and r.get("area_m2") is not None
    ]
    areas += [e["area_m2"] for e in payload.get("extra_areas", [])]
    return areas


def find_raw(plan_id: str, approach: str) -> dict | None:
    slug = SLUG[approach]
    for path in RAW_DIR.glob(f"{plan_id}_{slug}-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if "text" in entry:
            return entry
    return None


def predicted_areas(plan_id: str, approach: str, tokens: list[OcrToken]) -> list[float] | None:
    refs = TokenRef.from_tokens(tokens)
    if approach == "rules":
        outcome = extract_rules(tokens)
        return [r.area_m2 for r in outcome.rooms if r.area_m2 is not None]
    entry = find_raw(plan_id, approach)
    if entry is None:
        return None
    try:
        extraction = PlanExtraction.model_validate(json.loads(entry["text"]))
    except Exception:  # noqa: BLE001 - a malformed body means no prediction, not a crash
        return None
    grounded = GROUND[approach](extraction, refs)
    outcome = ExtractionOutcome(SOURCE[approach], grounded, None, refs)
    return [r.area_m2 for r in outcome.rooms if r.area_m2 is not None]


def match(predicted: list[float], gold: list[float], tolerance: float) -> int:
    """Greedy one-to-one match. Each gold area can be claimed once."""
    remaining = list(gold)
    hits = 0
    for value in predicted:
        for i, target in enumerate(remaining):
            if abs(value - target) <= tolerance * max(abs(target), 1e-6):
                hits += 1
                remaining.pop(i)
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="relative tolerance for an area match (default 5%%)")
    parser.add_argument("--list", type=Path, default=GOLD_LIST)
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/tier4_area_accuracy.md"))
    args = parser.parse_args()

    plan_ids = read_gold_list(args.list)
    ready: list[str] = []
    not_ready: list[str] = []
    for plan_id in plan_ids:
        (ready if gold_areas(plan_id) is not None else not_ready).append(plan_id)

    if not ready:
        print("No completed gold plans yet. Label some first:")
        print("  docker compose run --rm --no-deps api python /eval/label_helper.py")
        return 1

    totals = {a: {"pred": 0, "gold": 0, "hit": 0, "plans": 0} for a in APPROACHES}
    per_plan: dict[str, dict[str, tuple[int, int, int]]] = {}
    missing: list[str] = []

    for plan_id in ready:
        gold = gold_areas(plan_id) or []
        tokens = load_tokens(plan_id)
        if tokens is None:
            missing.append(f"{plan_id}: no OCR cache")
            continue
        per_plan[plan_id] = {}
        for approach in APPROACHES:
            predicted = predicted_areas(plan_id, approach, tokens)
            if predicted is None:
                missing.append(f"{plan_id}/{approach}: no cached result")
                continue
            hits = match(predicted, gold, args.tolerance)
            per_plan[plan_id][approach] = (hits, len(predicted), len(gold))
            totals[approach]["hit"] += hits
            totals[approach]["pred"] += len(predicted)
            totals[approach]["gold"] += len(gold)
            totals[approach]["plans"] += 1

    n = len(per_plan)
    lines = [
        "# Tier 4 - area accuracy against the hand-verified gold set",
        "",
        f"- Ground truth: **hand-verified** (Tier 4), n = {n} plans",
        f"- Match tolerance: {args.tolerance:.0%} relative, greedy one-to-one",
        "- Computed entirely from cache. No API calls were made.",
        "",
    ]
    if not_ready:
        lines.append(
            f"> {len(not_ready)} of {len(plan_ids)} gold plans are not finished and are "
            f"excluded: {', '.join(not_ready)}\n"
        )
    lines.append("| Approach | Precision | Recall | Found | Reported | Printed |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for approach in APPROACHES:
        t = totals[approach]
        if not t["plans"]:
            lines.append(f"| `{approach}` | no cached results | | | | |")
            continue
        precision = t["hit"] / t["pred"] if t["pred"] else 0.0
        recall = t["hit"] / t["gold"] if t["gold"] else 0.0
        lines.append(
            f"| `{approach}` | {precision:.0%} | {recall:.0%} | {t['hit']} | "
            f"{t['pred']} | {t['gold']} |"
        )
    lines.append("")
    lines.append("## Per plan (found / reported, printed)")
    lines.append("")
    lines.append("| Plan | Printed | " + " | ".join(f"`{a}`" for a in APPROACHES) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in APPROACHES) + " |")
    for plan_id in sorted(per_plan, key=int):
        row = per_plan[plan_id]
        printed = next(iter(row.values()))[2] if row else 0
        cells = []
        for approach in APPROACHES:
            if approach not in row:
                cells.append("-")
                continue
            hits, pred, _ = row[approach]
            cells.append(f"{hits}/{pred}")
        lines.append(f"| {plan_id} | {printed} | " + " | ".join(cells) + " |")
    lines.append("")
    if missing:
        lines.append("## Missing inputs\n")
        lines.extend(f"- {m}" for m in missing)
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"gold plans scored: {n} (of {len(plan_ids)} in the list)")
    print(f"\n{'approach':<10} {'precision':>10} {'recall':>8} {'found':>7} "
          f"{'reported':>9} {'printed':>8}")
    for approach in APPROACHES:
        t = totals[approach]
        if not t["plans"]:
            print(f"{approach:<10} {'no cached results':>10}")
            continue
        precision = t["hit"] / t["pred"] if t["pred"] else 0.0
        recall = t["hit"] / t["gold"] if t["gold"] else 0.0
        print(f"{approach:<10} {precision:>9.0%} {recall:>8.0%} {t['hit']:>7} "
              f"{t['pred']:>9} {t['gold']:>8}")
    if not_ready:
        print(f"\nnot finished, excluded: {', '.join(not_ready)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
