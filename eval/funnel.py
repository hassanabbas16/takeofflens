"""Area-recall funnel: find out WHERE printed areas are lost.

Area recall sits at roughly a quarter of the areas actually printed. That is a single
number hiding several very different failure modes, and the fix is completely different
depending on which stage is leaking. This walks each printed area through the pipeline:

    printed on the page
      -> OCR detected a box there
      -> OCR read the digits correctly
      -> parse_dims classified it as an AREA
      -> pairing associated it with a room label
      -> the approach reported it

Ground truth is the hand-read list of areas actually printed on each drawing, recorded in
ground_truth/printed_areas_6.json. Only the two plans that print per-room areas can be
measured - the other four print none, which is itself the finding.

Runs entirely from the OCR cache. No API calls, no cost.

    python eval/funnel.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef, find_candidate_pairs
from app.pipeline.parse_dims import DimKind, parse

PRINTED = Path(__file__).parent / "ground_truth" / "printed_areas_6.json"
OCR_CACHE = Path("/eval/cache/ocr")

# Tolerance for "this token is that printed area". Tight, because we are matching an OCR
# reading against a number we read off the page by eye, not against a derived polygon area.
TOL = 0.005


def printed_areas(notes: str) -> list[float]:
    """Pull the area values out of the hand-written note for a plan.

    The notes read like "MH 11.7 x2, K 14.5, ..." - "x2" means the same value twice.
    """
    values: list[float] = []
    for chunk in notes.split(","):
        chunk = chunk.strip()
        numbers = re.findall(r"\d+\.\d+", chunk)
        if not numbers:
            continue
        repeat = re.search(r"\bx(\d+)\b", chunk)
        times = int(repeat.group(1)) if repeat else 1
        values.extend([float(numbers[0])] * times)
    return values


def load_tokens(plan_id: str) -> list[OcrToken] | None:
    path = OCR_CACHE / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        OcrToken(text=t["text"], confidence=t["confidence"],
                 bbox=tuple(t["bbox"]), angle=t["angle"])
        for t in data["tokens"]
    ]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(abs(b), 1e-6)


def main() -> int:
    plans = json.loads(PRINTED.read_text(encoding="utf-8"))["plans"]

    stages = {
        "printed on the page": 0,
        "OCR read the digits": 0,
        "parser called it an AREA": 0,
        "paired with a label": 0,
    }
    missing_detail: list[str] = []

    print("=" * 78)
    print("AREA RECALL FUNNEL (from the OCR cache - no API calls)")
    print("=" * 78)

    for plan_id, info in plans.items():
        if not info["printed_area_count"]:
            continue
        expected = printed_areas(info["notes"])
        tokens = load_tokens(plan_id)
        if tokens is None:
            print(f"{plan_id}: no OCR cache entry, skipping")
            continue

        refs = TokenRef.from_tokens(tokens)
        pairs = find_candidate_pairs(refs)
        paired_values = [p.area_m2 for p in pairs]

        # Every number OCR produced, whatever the parser later made of it.
        ocr_numbers: list[float] = []
        for ref in refs:
            for raw in re.findall(r"\d+[.,]\d+", ref.text):
                try:
                    ocr_numbers.append(float(raw.replace(",", ".")))
                except ValueError:
                    continue

        parsed_areas = [
            parse(r.text).area_m2
            for r in refs
            if parse(r.text).kind is DimKind.AREA and parse(r.text).area_m2 is not None
        ]

        print(f"\n{plan_id}  ({info['label_style']})")
        print(f"  printed areas: {len(expected)}  ->  {expected}")

        read = [v for v in expected if any(close(v, n) for n in ocr_numbers)]
        as_area = [v for v in expected if any(close(v, n) for n in parsed_areas)]
        paired = [v for v in expected if any(close(v, n) for n in paired_values)]

        stages["printed on the page"] += len(expected)
        stages["OCR read the digits"] += len(read)
        stages["parser called it an AREA"] += len(as_area)
        stages["paired with a label"] += len(paired)

        print(f"  OCR read the digits:      {len(read):>2}/{len(expected)}  "
              f"missing {sorted(set(expected) - set(read))}")
        print(f"  parser called it an AREA: {len(as_area):>2}/{len(expected)}  "
              f"lost here {sorted(set(read) - set(as_area))}")
        print(f"  paired with a label:      {len(paired):>2}/{len(expected)}  "
              f"lost here {sorted(set(as_area) - set(paired))}")

        for value in sorted(set(as_area) - set(paired)):
            for ref in refs:
                result = parse(ref.text)
                if result.area_m2 is not None and close(value, result.area_m2):
                    missing_detail.append(
                        f"  {plan_id}: area {value:g} read as {ref.text!r} at {ref.bbox} "
                        "but no room label was near enough to pair with"
                    )
                    break

    print("\n" + "=" * 78)
    print("FUNNEL TOTALS (the 2 plans that print areas)")
    print("=" * 78)
    start = stages["printed on the page"]
    previous = start
    for name, count in stages.items():
        pct = 100 * count / max(start, 1)
        drop = previous - count
        bar = "#" * int(pct / 2.5)
        print(f"  {name:<26} {count:>3}/{start}  {pct:5.1f}%  {bar}"
              + (f"   (-{drop} here)" if drop else ""))
        previous = count

    if missing_detail:
        print("\nAreas the parser found but pairing dropped:")
        for line in missing_detail:
            print(line)

    print("\nStages beyond pairing (what each approach reported) are in "
          "eval/results/tier3_cost_probe.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
