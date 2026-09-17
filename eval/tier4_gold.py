"""Tier 4 step 1: choose the 5 plans that will be hand-labelled.

The gold set is no longer the primary area metric - that is now the SVG polygon, over all 50
plans (``eval/tier2_area_accuracy.py``). These 5 plans exist to **validate** that method: to
measure how often the figure printed on the drawing falls outside 5% of the polygon area, so
the polygon's limitation is a measured number rather than a caveat.

That is a much smaller job than scoring accuracy, which is why 5 plans is enough: it needs a
sample spanning the ways plans vary, not statistical power over every room in the split.

Drawn from the **50-plan Tier 3 sample**, so the validation applies to exactly the plans the
main table is computed over.

Selection is a documented, reproducible rule, not a hand-pick. Every chosen plan is written
to ``eval/data/gold_15.txt`` with the reason it was chosen, and every rejected plan with the
reason it was not, so the set is auditable.

What "readable areas" means here
--------------------------------
The Tier 1 signal for these plans is the cached OCR. A plan is a candidate only if the
drawing appears to print areas at all, judged from two counts:

``parser_areas``
    tokens the dimension parser reads as a plausible area. These are the clean cases.

``garbled_areas``
    tokens that hold a decimal, or an integer with a damaged area unit, but that the parser
    refuses. PaddleOCR renders the superscript in "m2" as LaTeX-like noise ("3,3 m^{2}$"),
    and a human reading the page sees the area perfectly well. These are kept and in fact
    *preferred*, because they are exactly where the four approaches disagree - a gold set of
    only clean plans would score every approach the same and measure nothing.

A plan with neither has no printed areas to verify and cannot support an area metric, so it
is excluded rather than labelled as all-nulls.

    python eval/tier4_gold.py
    python eval/tier4_gold.py --count 15 --out /eval/data/gold_15.txt

No API calls. Free.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.parse_dims import DimKind, parse

from svg_ground_truth import parse_model_svg, plan_dir

OCR_CACHE_DIR = Path("/eval/cache/ocr")
RESULTS_JSON = Path("/eval/results/tier3_sample50.json")

# A decimal, or an integer carrying a damaged area unit - the two forms a human can read as
# an area even when the parser cannot. Mirrors the grounding check's transcription rule.
_DECIMAL = re.compile(r"\d+[.,]\d+")
_INT_WITH_UNIT = re.compile(r"(\d+)\s*[mM]\s*[\^{\s]{0,3}[2²]")

# Below this many readable area candidates a plan cannot support an area metric.
MIN_READABLE_AREAS = 3


@dataclass
class Candidate:
    plan_id: str
    svg_rooms: int
    parser_areas: int
    garbled_areas: int
    tokens: int

    @property
    def readable_areas(self) -> int:
        return self.parser_areas + self.garbled_areas

    @property
    def eligible(self) -> bool:
        return self.readable_areas >= MIN_READABLE_AREAS and self.svg_rooms > 0


def count_area_evidence(plan_id: str) -> tuple[int, int, int] | None:
    path = OCR_CACHE_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tokens" not in data:
        return None
    parser_areas = garbled = 0
    for token in data["tokens"]:
        text = token["text"]
        result = parse(text)
        if result.kind is DimKind.AREA and result.area_m2 is not None and result.plausible:
            parser_areas += 1
        elif _DECIMAL.search(text) or _INT_WITH_UNIT.search(text):
            garbled += 1
    return parser_areas, garbled, len(data["tokens"])


def sample_plan_ids() -> list[str]:
    """The 50 Tier 3 plans, read from the run's own output so the two cannot drift."""
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return sorted({row["plan"] for row in data["results"]["rules"]}, key=int)


def choose(candidates: list[Candidate], count: int) -> list[tuple[Candidate, str]]:
    """Spread the pick across room counts, preferring plans with garbled areas.

    With only 5 plans to label, variety matters more than volume - the set has to span the
    ways these drawings differ, or it will validate the polygon method against one house
    style. Eligible plans are split into room-count bands and taken round-robin, so a small
    flat and a large house both appear. Within a band, plans whose area text OCR garbled come
    first: those exercise the repair and the pairing hardest.
    """
    eligible = [c for c in candidates if c.eligible]

    def band(c: Candidate) -> str:
        if c.svg_rooms <= 4:
            return "small (<=4 rooms)"
        if c.svg_rooms <= 8:
            return "medium (5-8 rooms)"
        return "large (9+ rooms)"

    bands: dict[str, list[Candidate]] = {}
    for c in eligible:
        bands.setdefault(band(c), []).append(c)
    for members in bands.values():
        members.sort(key=lambda c: (-c.garbled_areas, -c.readable_areas, int(c.plan_id)))

    chosen: list[tuple[Candidate, str]] = []
    order = sorted(bands)
    while len(chosen) < count and any(bands[b] for b in order):
        for b in order:
            if not bands[b] or len(chosen) >= count:
                continue
            c = bands[b].pop(0)
            reason = (
                f"{b}; {c.svg_rooms} SVG rooms; "
                f"{c.parser_areas} parser-readable + {c.garbled_areas} garbled-readable areas"
            )
            chosen.append((c, reason))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    parser.add_argument("--out", type=Path, default=Path("/eval/data/gold_5.txt"))
    args = parser.parse_args()

    candidates: list[Candidate] = []
    skipped: list[tuple[str, str]] = []
    for plan_id in sample_plan_ids():
        evidence = count_area_evidence(plan_id)
        if evidence is None:
            skipped.append((plan_id, "no OCR cache entry"))
            continue
        parser_areas, garbled, n_tokens = evidence
        directory = plan_dir(args.dataset, f"/high_quality_architectural/{plan_id}/")
        try:
            svg = parse_model_svg(directory / "model.svg")
        except Exception as exc:  # noqa: BLE001 - a bad SVG is a skip, not a crash
            skipped.append((plan_id, f"unparseable model.svg: {exc}"))
            continue
        candidates.append(
            Candidate(plan_id, svg.room_count, parser_areas, garbled, n_tokens)
        )

    chosen = choose(candidates, args.count)
    chosen_ids = {c.plan_id for c, _ in chosen}
    for c in sorted(candidates, key=lambda c: int(c.plan_id)):
        if c.plan_id in chosen_ids:
            continue
        if not c.eligible:
            skipped.append((
                c.plan_id,
                f"only {c.readable_areas} readable area token(s), need {MIN_READABLE_AREAS}"
                if c.svg_rooms else "no rooms in model.svg",
            ))
        else:
            skipped.append((c.plan_id, "eligible but not picked (band already filled)"))

    lines = [
        "# Tier 4 gold set: 5 hand-labelled plans",
        "#",
        "# Chosen by eval/tier4_gold.py from the 50-plan Tier 3 sample (seed 20260917), so",
        "# every plan here already has cached results for all four approaches and area",
        f"# accuracy costs nothing to compute. Eligibility: >= {MIN_READABLE_AREAS} readable",
        "# area tokens in the cached OCR. Picked round-robin across room-count bands,",
        "# preferring plans whose areas OCR garbled - those are where the approaches differ.",
        "#",
        "# plan_id  reason",
    ]
    for c, reason in sorted(chosen, key=lambda t: int(t[0].plan_id)):
        lines.append(f"{c.plan_id}  # {reason}")
    lines.append("")
    lines.append("# --- not chosen ---")
    for plan_id, reason in sorted(skipped, key=lambda t: int(t[0])):
        lines.append(f"# {plan_id}  {reason}")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    # Report which bands actually had anything in them. The round-robin can only spread
    # across bands that are occupied, and on this dataset the small band is empty: a plan
    # with four rooms or fewer does not print enough areas to clear the threshold. Better
    # to say so than to present a large-skewed set as if it were balanced.
    occupancy: dict[str, int] = {}
    for c in candidates:
        if not c.eligible:
            continue
        key = ("small (<=4 rooms)" if c.svg_rooms <= 4
               else "medium (5-8 rooms)" if c.svg_rooms <= 8 else "large (9+ rooms)")
        occupancy[key] = occupancy.get(key, 0) + 1
    lines.insert(8, "# Eligible plans per band: " + (
        ", ".join(f"{k} {v}" for k, v in sorted(occupancy.items())) or "none"
    ))
    lines.insert(9, "#")
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(candidates)} candidates, {sum(1 for c in candidates if c.eligible)} eligible")
    print("eligible per band: " + ", ".join(f"{k}={v}" for k, v in sorted(occupancy.items())))
    print(f"chose {len(chosen)}:\n")
    print(f"{'plan':>7} {'rooms':>6} {'parser':>7} {'garbled':>8}  band")
    for c, reason in sorted(chosen, key=lambda t: int(t[0].plan_id)):
        print(f"{c.plan_id:>7} {c.svg_rooms:>6} {c.parser_areas:>7} {c.garbled_areas:>8}  "
              f"{reason.split(';')[0]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
