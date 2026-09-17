"""Validate the polygon method: how far is the printed area from the polygon area?

The primary area metric (``eval/tier2_area_accuracy.py``) scores every approach against the
shoelace area of the CubiCasa annotation polygon, over all 50 sampled plans. That is free and
it covers every room, but the polygon is not what the drawing prints - it follows the inner
wall face, while the printed figure uses the agent's own convention.

This script quantifies that gap, and nothing else. It takes the 5 hand-labelled plans, pairs
each human-read printed area with its own room's polygon area, and reports the distribution
of (printed - polygon) / polygon.

The number that matters is **the share of printed areas falling outside 5% of the polygon**.
That is the rate at which the primary metric marks a correct reading wrong through no fault
of the extractor, and it is why that metric is also reported at 10%.

Ground truth here is hand-verified. It is used to characterise the automatic ground truth,
never to score an approach - with n=5 plans it could not support an accuracy claim, and this
file makes no such claim.

    python eval/tier4_gold_validation.py

No API calls. Free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.room_types import label_key

from svg_ground_truth import parse_model_svg, plan_dir

GOLD_DIR = Path("/eval/ground_truth/gold")
GOLD_LIST = Path("/eval/data/gold_5.txt")


def read_gold_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def printed_areas(plan_id: str) -> list[dict] | None:
    """Rooms the human decided on, with the SVG room each was pre-filled from."""
    path = GOLD_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        return None
    return payload["rooms"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    parser.add_argument("--list", type=Path, default=GOLD_LIST)
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/tier4_gold_validation.md"))
    args = parser.parse_args()

    plan_ids = read_gold_list(args.list)
    ready, not_ready = [], []
    for plan_id in plan_ids:
        (ready if printed_areas(plan_id) is not None else not_ready).append(plan_id)

    if not ready:
        print("No completed gold plans yet. Label them first:")
        print("  docker compose run --rm --no-deps api python /eval/label_helper.py")
        print(f"\nwaiting on: {', '.join(not_ready)}")
        return 1

    offsets: list[float] = []
    rows: list[str] = []
    no_area = 0
    for plan_id in ready:
        rooms = printed_areas(plan_id) or []
        svg = parse_model_svg(
            plan_dir(args.dataset, f"/high_quality_architectural/{plan_id}/") / "model.svg"
        )
        by_index = {i: r for i, r in enumerate(svg.rooms)}
        for room in rooms:
            if room.get("area_m2") is None:
                no_area += 1
                continue
            svg_room = by_index.get(room["svg_index"])
            if svg_room is None or svg_room.area_m2 <= 0:
                continue
            printed = room["area_m2"]
            polygon = svg_room.area_m2
            offset = (printed - polygon) / polygon
            offsets.append(offset)
            rows.append(
                f"| {plan_id} | {label_key(room.get('label') or room['svg_name'])} | "
                f"{printed:.1f} | {polygon:.2f} | {offset:+.1%} | "
                f"{'yes' if abs(offset) <= args.tolerance else '**no**'} |"
            )

    if not offsets:
        print("No printed areas recorded on the completed plans - nothing to compare.")
        return 1

    outside = [o for o in offsets if abs(o) > args.tolerance]
    outside10 = [o for o in offsets if abs(o) > 0.10]
    median = statistics.median(offsets)
    mean = statistics.fmean(offsets)

    lines = [
        "# Validating the polygon method against the drawing",
        "",
        f"- Hand-verified plans used: **{len(ready)}** of {len(plan_ids)} "
        f"({', '.join(ready)})",
        f"- Printed areas compared: **{len(offsets)}**",
        f"- Rooms the human recorded as printing no area: {no_area}",
        "",
        "Every printed area is compared with **its own room's** polygon, paired by the SVG "
        "room the labelling CLI pre-filled it from - so this measures the convention gap, "
        "not a matching failure.",
        "",
        "## Result",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Median (printed - polygon) / polygon | **{median:+.1%}** |",
        f"| Mean | {mean:+.1%} |",
        f"| Printed areas **outside {args.tolerance:.0%}** of the polygon | "
        f"**{len(outside)}/{len(offsets)} ({len(outside) / len(offsets):.0%})** |",
        f"| Printed areas outside 10% | {len(outside10)}/{len(offsets)} "
        f"({len(outside10) / len(offsets):.0%}) |",
        "",
        f"**{len(outside) / len(offsets):.0%} of correctly-read areas would be scored wrong "
        f"at {args.tolerance:.0%}** against polygon ground truth, through no fault of the "
        "extractor. That is the headline limitation of the primary metric, and the reason "
        "`tier2_area_accuracy.md` reports 10% alongside 5%.",
        "",
        "The polygon follows the inner wall face. A positive median means the printed figure "
        "is the larger of the two, which is consistent with the drawing quoting an area "
        "measured to something other than the inner face.",
        "",
        "## Every comparison",
        "",
        "| Plan | Room | Printed | Polygon | Offset | Within tolerance |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    if not_ready:
        lines += [
            f"> {len(not_ready)} of {len(plan_ids)} gold plans are not labelled yet and are "
            f"excluded: {', '.join(not_ready)}",
            "",
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(ready)} plans, {len(offsets)} printed areas compared")
    print(f"  median offset  {median:+.1%}")
    print(f"  mean offset    {mean:+.1%}")
    print(f"  outside {args.tolerance:.0%}      {len(outside)}/{len(offsets)} "
          f"({len(outside) / len(offsets):.0%})")
    print(f"  outside 10%     {len(outside10)}/{len(offsets)} "
          f"({len(outside10) / len(offsets):.0%})")
    if not_ready:
        print(f"\nnot labelled yet: {', '.join(not_ready)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
