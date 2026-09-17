"""Area accuracy for all four approaches over all 50 sampled plans, referenced to model.svg.

**Automatic ground truth, not hand-verified.** Every number here is measured against the
shoelace area of the CubiCasa annotation polygon, and the README and every table say so.

Why this replaces the hand-labelled path as the primary area metric
-------------------------------------------------------------------
Hand-labelling 15 plans is expensive and yields n=15. The SVG polygon gives a room-level area
for all 50 plans and every room on them, at no cost, which is 736 rooms rather than a few
dozen. The price is that a polygon area is not the figure printed on the drawing: the polygon
follows the inner wall face and the printed figure uses the agent's own convention, so they
disagree by a small percentage. That disagreement is the method's limitation, and it is
**measured rather than assumed** - see ``eval/tier4_gold_validation.py``, which uses a 5-plan
hand-labelled set for exactly this and nothing else.

Categories are defined in ``eval/area_scoring.py``. Runs entirely from cache: `rules` is
recomputed (free, deterministic) and the three paid approaches are replayed from the raw
batch bodies. **No API calls.**

    python eval/tier2_area_accuracy.py
    python eval/tier2_area_accuracy.py --tolerance 0.10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline import parse_dims
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

from area_scoring import AreaScore, score_plan
from svg_ground_truth import parse_model_svg, plan_dir

OCR_CACHE_DIR = Path("/eval/cache/ocr")
RAW_DIR = Path("/eval/cache/batch_raw")
RESULTS_JSON = Path("/eval/results/tier3_sample50.json")

APPROACHES = ["rules", "ocr+llm", "vlm", "hybrid"]
SLUG = {"ocr+llm": "ocr_llm", "vlm": "vlm", "hybrid": "hybrid"}
GROUND = {"ocr+llm": ground_ocr_llm, "vlm": ground_vlm, "hybrid": ground_hybrid}
SOURCE = {"ocr+llm": Source.OCR_LLM, "vlm": Source.VLM, "hybrid": Source.HYBRID}

# Rooms smaller than this are cupboards and slivers the drawing never labels with an area.
MIN_GOLD_AREA_M2 = 1.0


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


def sample_plan_ids() -> list[str]:
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return sorted({row["plan"] for row in data["results"]["rules"]}, key=int)


def svg_rooms(plan_id: str, dataset: Path) -> list[tuple[str, float]] | None:
    directory = plan_dir(dataset, f"/high_quality_architectural/{plan_id}/")
    try:
        plan = parse_model_svg(directory / "model.svg")
    except Exception:  # noqa: BLE001 - an unreadable SVG is a skip, not a crash
        return None
    if not plan.scale_ok:
        return None
    return [
        (r.name or "", r.area_m2)
        for r in plan.rooms
        if r.area_m2 >= MIN_GOLD_AREA_M2 and (r.name or "").strip()
    ]


def find_raw(plan_id: str, approach: str) -> dict | None:
    for path in RAW_DIR.glob(f"{plan_id}_{SLUG[approach]}-*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if "text" in entry:
            return entry
    return None


def predictions(
    plan_id: str, approach: str, tokens: list[OcrToken]
) -> list[tuple[str, float]] | None:
    refs = TokenRef.from_tokens(tokens)
    if approach == "rules":
        outcome = extract_rules(tokens)
    else:
        entry = find_raw(plan_id, approach)
        if entry is None:
            return None
        try:
            extraction = PlanExtraction.model_validate(json.loads(entry["text"]))
        except Exception:  # noqa: BLE001
            return None
        grounded = GROUND[approach](extraction, refs)
        outcome = ExtractionOutcome(SOURCE[approach], grounded, None, refs)
    return [
        (r.label_raw or "", r.area_m2) for r in outcome.rooms if r.area_m2 is not None
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/tier2_area_accuracy.md"))
    parser.add_argument("--regression-plan", default="416",
                        help="plan whose extraction was verified by eye, used to check that "
                             "this metric moves when the pipeline really improves")
    args = parser.parse_args()

    plan_ids = sample_plan_ids()
    totals = {a: AreaScore() for a in APPROACHES}
    examples: dict[str, list[str]] = {a: [] for a in APPROACHES}
    per_plan: dict[str, dict[str, AreaScore]] = {}
    skipped: list[str] = []

    for plan_id in plan_ids:
        tokens = load_tokens(plan_id)
        gold = svg_rooms(plan_id, args.dataset)
        if tokens is None or gold is None:
            skipped.append(plan_id)
            continue
        per_plan[plan_id] = {}
        for approach in APPROACHES:
            pred = predictions(plan_id, approach, tokens)
            if pred is None:
                continue
            score = score_plan(
                pred, gold, args.tolerance, plan_id=plan_id, collect=3
            )
            totals[approach].add(score)
            per_plan[plan_id][approach] = score
            for example in score.examples:
                if len(examples[approach]) < 6:
                    examples[approach].append(example)

    # A second full pass at 10%, reported beside the 5% table rather than buried in the
    # sensitivity sweep: at 5% the metric is dominated by the convention offset.
    wide = {a: AreaScore() for a in APPROACHES}
    for plan_id in per_plan:
        tokens = load_tokens(plan_id)
        gold = svg_rooms(plan_id, args.dataset)
        if tokens is None or gold is None:
            continue
        for approach in APPROACHES:
            pred = predictions(plan_id, approach, tokens)
            if pred is not None:
                wide[approach].add(score_plan(pred, gold, 0.10))

    # Tolerance sensitivity, because 5% is a choice and the reader should see its effect.
    sweep: dict[float, dict[str, int]] = {}
    for tol in (0.02, 0.05, 0.10, 0.20):
        sweep[tol] = {}
        for approach in APPROACHES:
            agg = AreaScore()
            for plan_id in per_plan:
                tokens = load_tokens(plan_id)
                gold = svg_rooms(plan_id, args.dataset)
                if tokens is None or gold is None:
                    continue
                pred = predictions(plan_id, approach, tokens)
                if pred is None:
                    continue
                agg.add(score_plan(pred, gold, tol))
            sweep[tol][approach] = agg.correct

    # Does this metric actually track a real improvement? The OCR area-unit repair is a
    # known-good change whose effect on plan 416 was checked against the drawing by eye
    # (3 of 6 correct -> 8 of 9). A metric that cannot see it is not measuring the pipeline.
    regression: list[tuple[float, int, int]] = []
    reg_tokens = load_tokens(args.regression_plan)
    reg_gold = svg_rooms(args.regression_plan, args.dataset)
    if reg_tokens is not None and reg_gold is not None:
        real = parse_dims.normalise_area_unit
        for tol in (0.05, 0.10):
            counts = []
            for fn in (lambda text: text, real):
                parse_dims.normalise_area_unit = fn
                try:
                    pred = predictions(args.regression_plan, "rules", reg_tokens)
                finally:
                    parse_dims.normalise_area_unit = real
                counts.append(score_plan(pred or [], reg_gold, tol).correct)
            regression.append((tol, counts[0], counts[1]))

    n = len(per_plan)
    rooms = totals["rules"].gold_rooms
    offsets = sorted(totals["rules"].offsets)
    median_off = statistics.median(offsets) if offsets else 0.0
    within5 = sum(1 for o in offsets if abs(o) <= 0.05) / len(offsets) if offsets else 0.0
    within10 = sum(1 for o in offsets if abs(o) <= 0.10) / len(offsets) if offsets else 0.0
    lines = [
        "# Area accuracy vs the `model.svg` polygons - all 50 sampled plans",
        "",
        "- Ground truth: **automatic (Tier 2)**, the shoelace area of the CubiCasa "
        "annotation polygon. Not hand-verified.",
        f"- {n} plans, {rooms} annotated rooms of at least {MIN_GOLD_AREA_M2:g} m2 with a name",
        f"- Match tolerance: **{args.tolerance:.0%}**, on room **and** area together",
        "- Computed entirely from cache. No API calls.",
        "",
        "> **What a polygon area is and is not.** It is the area CubiCasa's annotator drew,",
        "> following the inner wall face. It is *not* the figure printed on the drawing,",
        "> which uses the agent's own convention. The two disagree by a few percent, so a",
        "> correct reading can still be scored wrong here. How often that happens is measured",
        "> on a 5-plan hand-labelled set - see `tier4_gold_validation.md` - rather than",
        "> assumed. Treat these as comparative numbers between approaches, which is what they",
        "> are good for, not as absolute accuracy against the page.",
        "",
        "## The polygon is systematically smaller than the printed figure",
        "",
        "Measured over every label-matched `rules` pair on these plans "
        f"(n = {len(offsets)}), as (extracted - polygon) / polygon:",
        "",
        f"- median offset **{median_off:+.1%}**",
        f"- within 5% of the polygon: **{within5:.0%}** of pairs",
        f"- within 10%: **{within10:.0%}** of pairs",
        "",
        "The polygon follows the inner wall face; the printed figure does not. On plan 416, "
        "where every extracted value was checked against the drawing by eye, the correct "
        "readings sit +8% to +13% from their polygons - MH 12.3 vs 11.24, TH 7.4 vs 6.83, "
        "KHH 6.0 vs 5.46. **A 5% band cannot contain a systematic offset of that size**, so "
        "at 5% this metric scores correct readings as wrong.",
        "",
        "That is a property of the ground truth, not of the matcher, which is why both "
        "tolerances are reported. Quantifying it against the drawing is the one job the "
        "5-plan hand-labelled set exists to do.",
        "",
        "## Does this metric track a real improvement?",
        "",
        f"The OCR area-unit repair is a known-good change. On plan {args.regression_plan}, "
        "every extracted value was checked against the drawing by eye: the repair took it "
        "from 3 of 6 correct to 8 of 9, and corrected two already-wrong values. `rules` "
        "correct on that plan, before and after the repair:",
        "",
        "| Tolerance | Before | After |",
        "| --- | --- | --- |",
        *[f"| {tol:.0%} | {b} | {a} |" for tol, b, a in regression],
        "",
        "**At 5% the metric cannot see the improvement at all; at 10% it does.** That is the "
        "convention offset, not the matcher: room pairing is working, and it is what stops "
        "the older area-only matcher from scoring a wrong kitchen value as correct by "
        "accidentally matching it against an unrelated room. The 10% row is the one that "
        "reflects the pipeline.",
        "",
        "## Results at 5%",
        "",
        "| Approach | Precision | Recall | Correct | Wrong value | Misattributed | "
        "Hallucinated | Reported |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for approach in APPROACHES:
        t = totals[approach]
        lines.append(
            f"| `{approach}` | {t.precision:.0%} | {t.recall:.0%} | {t.correct} | "
            f"{t.wrong_value} | {t.misattributed} | {t.hallucinated} | {t.reported} |"
        )
    lines += [
        "",
        "## Results at 10%",
        "",
        f"The tolerance that actually accommodates the {median_off:+.0%} convention offset:",
        "",
        "| Approach | Precision | Recall | Correct | Wrong value | Misattributed | "
        "Hallucinated | Reported |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for approach in APPROACHES:
        t = wide[approach]
        lines.append(
            f"| `{approach}` | {t.precision:.0%} | {t.recall:.0%} | {t.correct} | "
            f"{t.wrong_value} | {t.misattributed} | {t.hallucinated} | {t.reported} |"
        )
    lines += [
        "",
        f"**Recall** is against all {rooms} annotated rooms. That denominator is a floor, not "
        "a target: many of those rooms have no area printed on the drawing at all, so no "
        "approach could reach 100% by reading the page correctly. It is comparable between "
        "approaches, which is the point.",
        "",
        "## Tolerance sensitivity",
        "",
        "5% is a choice. Correct counts at other tolerances:",
        "",
        "| Tolerance | " + " | ".join(f"`{a}`" for a in APPROACHES) + " |",
        "| --- | " + " | ".join("---" for _ in APPROACHES) + " |",
    ]
    for tol in (0.02, 0.05, 0.10, 0.20):
        lines.append(
            f"| {tol:.0%} | " + " | ".join(str(sweep[tol][a]) for a in APPROACHES) + " |"
        )
    lines += [
        "",
        "The gap between 5% and 10% is mostly the polygon-vs-printed convention difference, "
        "not extraction error - which is the limitation the gold set exists to quantify.",
        "",
        "## Failure examples",
        "",
    ]
    for approach in APPROACHES:
        lines.append(f"**`{approach}`**")
        lines.append("")
        for example in examples[approach] or ["- (none collected)"]:
            lines.append(f"- {example}")
        lines.append("")
    if skipped:
        lines.append(f"Skipped {len(skipped)} plans (no OCR cache or unusable SVG scale): "
                     f"{', '.join(skipped)}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"{n} plans, {rooms} annotated rooms, tolerance {args.tolerance:.0%}\n")
    print(f"{'approach':<10} {'prec':>6} {'recall':>7} {'correct':>8} {'wrong':>7} "
          f"{'misattr':>8} {'halluc':>7} {'reported':>9}")
    for approach in APPROACHES:
        t = totals[approach]
        print(f"{approach:<10} {t.precision:>5.0%} {t.recall:>7.0%} {t.correct:>8} "
              f"{t.wrong_value:>7} {t.misattributed:>8} {t.hallucinated:>7} "
              f"{t.reported:>9}")
    print(f"\npolygon-vs-extracted offset (n={len(offsets)}): median {median_off:+.1%}, "
          f"{within5:.0%} of pairs within 5%, {within10:.0%} within 10%")
    print(f"\n{'approach':<10} {'prec@10%':>9} {'recall@10%':>11} {'correct@10%':>12}")
    for approach in APPROACHES:
        t = wide[approach]
        print(f"{approach:<10} {t.precision:>8.0%} {t.recall:>10.0%} {t.correct:>12}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
