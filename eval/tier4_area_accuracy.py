"""Tier 4 step 3: area accuracy for all four approaches against the hand-verified gold set.

Runs entirely from cache - the gold JSON written by ``label_helper.py``, the OCR cache, and
the raw batch bodies saved by ``retrieve_batch.py``. **No API calls, no cost.** That is why
the gold plans were drawn from the 50-plan Tier 3 sample: their results are already paid for.

Reports, per approach, over the gold plans only:

precision
    of the (room, area) pairs an approach reported, how many are right on both counts.

recall
    of the areas actually printed, how many the approach attached to the right room.

misattributed
    the area is genuinely printed on the page, but the approach put it against the wrong
    room. Reported in its own column, and never folded into precision or into the
    hallucination counts: reading the page correctly and then assigning the value wrongly is
    a different defect from inventing a number, and it points at the bbox pairing rather
    than at the extractor.

A prediction counts as correct only when **room and area both match**. An area alone is not
a takeoff - a tool that reports the right number against the wrong room produces a quantity
survey that does not add up.

Precision and recall are reported separately and never averaged, because they fail in
opposite directions: `rules` is conservative and `vlm` is liberal, and a single score would
hide that. Matching is greedy within a relative tolerance and each gold entry can be claimed
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
from app.pipeline.room_types import label_key
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


def gold_areas(plan_id: str) -> list[tuple[str, float]] | None:
    """(label, printed area) pairs the human verified. None if the plan is not finished."""
    path = GOLD_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        return None
    pairs = [
        (label_key(r.get("label") or r.get("svg_name") or ""), r["area_m2"])
        for r in payload["rooms"]
        if r.get("decided") and r.get("area_m2") is not None
    ]
    pairs += [
        (label_key(e.get("label") or ""), e["area_m2"])
        for e in payload.get("extra_areas", [])
    ]
    return pairs


def find_raw(plan_id: str, approach: str) -> dict | None:
    slug = SLUG[approach]
    for path in RAW_DIR.glob(f"{plan_id}_{slug}-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if "text" in entry:
            return entry
    return None


def predicted_areas(
    plan_id: str, approach: str, tokens: list[OcrToken]
) -> list[tuple[str, float]] | None:
    """(label, area) pairs an approach reported. Both halves matter - see match()."""
    refs = TokenRef.from_tokens(tokens)
    if approach == "rules":
        outcome = extract_rules(tokens)
    else:
        entry = find_raw(plan_id, approach)
        if entry is None:
            return None
        try:
            extraction = PlanExtraction.model_validate(json.loads(entry["text"]))
        except Exception:  # noqa: BLE001 - a malformed body means no prediction, not a crash
            return None
        grounded = GROUND[approach](extraction, refs)
        outcome = ExtractionOutcome(SOURCE[approach], grounded, None, refs)
    return [
        (label_key(r.label_raw or ""), r.area_m2)
        for r in outcome.rooms
        if r.area_m2 is not None
    ]


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(b), 1e-6)


def match(
    predicted: list[tuple[str, float]],
    gold: list[tuple[str, float]],
    tolerance: float,
) -> tuple[int, int, int]:
    """Match on room *and* area together. Returns (correct, misattributed, spurious).

    An area alone is not a takeoff. "11.7 appears somewhere on this page" is worth much less
    than "the bedroom is 11.7", and a tool that reports the right number against the wrong
    room produces a quantity survey that does not add up. So a prediction is only *correct*
    when the label matches too.

    A prediction whose area is genuinely printed on the page but sits under a different room
    is **misattributed**. That is a distinct failure from a hallucination and is reported in
    its own column: the model read the page correctly and then assigned the value wrongly,
    which points at the bbox pairing rather than at the extractor inventing things.

    Two passes so a label-correct match is never stolen by a label-wrong one: every exact
    (label, area) pair is claimed first, and only the leftovers are tested for
    misattribution. Each gold entry can be claimed once, so repeating an area cannot inflate
    the score. Duplicate labels (three MH rooms) match within the label, which is the most
    that is knowable - nothing distinguishes one MH from another here.
    """
    remaining = list(gold)

    # Pass 1: claim every prediction that is right on both counts. Gold entries are consumed
    # here and only here, so five copies of one area cannot score five times - recall stays
    # honest. Doing this before anything else stops a label-wrong prediction listed earlier
    # from stealing the gold entry that a label-correct one deserves.
    correct = 0
    leftover: list[tuple[str, float]] = []
    for label, value in predicted:
        for i, (gold_label, gold_value) in enumerate(remaining):
            if label == gold_label and _close(value, gold_value, tolerance):
                correct += 1
                remaining.pop(i)
                break
        else:
            leftover.append((label, value))

    # Pass 2: classify what is left against the *whole* gold set, claimed or not. The
    # question here is about the prediction, not about a gold slot: "is this a real area put
    # against the wrong room?" is true whether or not some other room already claimed it.
    misattributed = 0
    spurious = 0
    for label, value in leftover:
        printed_here = [g_label for g_label, g_value in gold if _close(value, g_value, tolerance)]
        if printed_here and label not in printed_here:
            misattributed += 1
        else:
            # Either the area is printed nowhere, or it is a duplicate claim on a room that
            # was already matched. Both are unsupported extra claims.
            spurious += 1
    return correct, misattributed, spurious


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

    totals = {
        a: {"pred": 0, "gold": 0, "hit": 0, "mis": 0, "spur": 0, "plans": 0}
        for a in APPROACHES
    }
    per_plan: dict[str, dict[str, tuple[int, int, int, int, int]]] = {}
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
            hits, mis, spur = match(predicted, gold, args.tolerance)
            per_plan[plan_id][approach] = (hits, mis, spur, len(predicted), len(gold))
            totals[approach]["hit"] += hits
            totals[approach]["mis"] += mis
            totals[approach]["spur"] += spur
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
    lines.append(
        "A prediction is **correct** only when the room label and the area both match. "
        "An area that is genuinely printed on the page but attached to the wrong room "
        "is **misattributed** and counted in its own column - the page was read "
        "correctly and the assignment was wrong, which is a different defect from "
        "inventing a number, and different again from a grounding failure (those are "
        "in `tier3_sample50.md`).\n"
    )
    lines.append(
        "| Approach | Precision | Recall | Correct | Misattributed | Spurious | "
        "Reported | Printed |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for approach in APPROACHES:
        t = totals[approach]
        if not t["plans"]:
            lines.append(f"| `{approach}` | no cached results | | | | | | |")
            continue
        precision = t["hit"] / t["pred"] if t["pred"] else 0.0
        recall = t["hit"] / t["gold"] if t["gold"] else 0.0
        lines.append(
            f"| `{approach}` | {precision:.0%} | {recall:.0%} | {t['hit']} | "
            f"{t['mis']} | {t['spur']} | {t['pred']} | {t['gold']} |"
        )
    lines.append("")
    lines.append("## Per plan - correct (+misattributed) / reported")
    lines.append("")
    lines.append("| Plan | Printed | " + " | ".join(f"`{a}`" for a in APPROACHES) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in APPROACHES) + " |")
    for plan_id in sorted(per_plan, key=int):
        row = per_plan[plan_id]
        printed = next(iter(row.values()))[4] if row else 0
        cells = []
        for approach in APPROACHES:
            if approach not in row:
                cells.append("-")
                continue
            hits, mis, _spur, pred, _gold = row[approach]
            cells.append(f"{hits}/{pred}" + (f" (+{mis} mis)" if mis else ""))
        lines.append(f"| {plan_id} | {printed} | " + " | ".join(cells) + " |")
    lines.append("")
    if missing:
        lines.append("## Missing inputs\n")
        lines.extend(f"- {m}" for m in missing)
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"gold plans scored: {n} (of {len(plan_ids)} in the list)")
    print(f"\n{'approach':<10} {'prec':>6} {'recall':>7} {'correct':>8} "
          f"{'misattr':>8} {'spurious':>9} {'reported':>9} {'printed':>8}")
    for approach in APPROACHES:
        t = totals[approach]
        if not t["plans"]:
            print(f"{approach:<10} {'no cached results':>10}")
            continue
        precision = t["hit"] / t["pred"] if t["pred"] else 0.0
        recall = t["hit"] / t["gold"] if t["gold"] else 0.0
        print(f"{approach:<10} {precision:>5.0%} {recall:>7.0%} {t['hit']:>8} "
              f"{t['mis']:>8} {t['spur']:>9} {t['pred']:>9} {t['gold']:>8}")
    if not_ready:
        print(f"\nnot finished, excluded: {', '.join(not_ready)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
