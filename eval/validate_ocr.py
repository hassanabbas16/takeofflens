"""Validate the OCR settings and the dedup fix on several plans, not just one.

Two questions:

1. Does the new vocabulary-based dedup beat confidence-only dedup?
   Both are applied to the *same* raw token set per plan, so the comparison is exact and
   costs one OCR pass rather than one per variant.

2. Are the three settings chosen on plan 1191 (four rotation passes, 1536 detection size,
   adaptive threshold off) still right on a wider set?
   Each is varied independently against the default, which needs its own OCR pass.

Reference labels and areas come from ``model.svg`` via Tier 2. Areas there are polygon
areas, within about 1-2% of the figure printed on the drawing, so area matching uses a 5%
tolerance - the same tolerance the eval spec uses.

    python eval/validate_ocr.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")

from app.pipeline.ocr import OcrToken, deduplicate, iou, run_ocr
from app.pipeline.parse_dims import DimKind, parse
from app.pipeline.preprocess import PreprocessConfig, preprocess
from app.pipeline.room_types import normalise_label

sys.path.insert(0, str(Path(__file__).parent))
from svg_ground_truth import parse_model_svg, plan_dir, read_split

AREA_TOLERANCE = 0.05

# Hand-counted areas actually printed on each drawing. model.svg lists every annotated room,
# but its dimension labels are display:none and never rendered, so the SVG room count is the
# wrong denominator for "did OCR find the area?" - it counts areas that were never printed.
PRINTED_AREAS_PATH = Path(__file__).parent / "ground_truth" / "printed_areas_6.json"


def confidence_only_dedup(tokens: list[OcrToken], iou_threshold: float) -> list[OcrToken]:
    """The previous behaviour: greedy NMS keeping the highest-confidence reading."""
    kept: list[OcrToken] = []
    for token in sorted(tokens, key=lambda t: t.confidence, reverse=True):
        if any(iou(token.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(token)
    return kept


def reference(svg_path: Path) -> tuple[set[str], list[float]]:
    """Distinct Finnish labels and the room areas printed for this plan."""
    plan = parse_model_svg(svg_path)
    labels = {normalise_label(r.name) for r in plan.rooms if r.name}
    labels.discard("")
    labels.discard("UNDEFINED")  # an annotation placeholder, not a printed label
    areas = [r.area_m2 for r in plan.rooms if r.area_m2 >= 1.0]
    return labels, areas


def count_labels(tokens: list[OcrToken], labels: set[str]) -> int:
    seen = {normalise_label(t.text) for t in tokens}
    # Also accept a label that arrived glued to its area, e.g. "MH 11.7".
    for t in tokens:
        result = parse(t.text)
        if result.label:
            seen.add(normalise_label(result.label))
    return sum(1 for label in labels if label in seen)


def count_areas(tokens: list[OcrToken], areas: list[float]) -> int:
    """Greedy match: each OCR area may satisfy at most one reference area."""
    found = []
    for t in tokens:
        result = parse(t.text)
        if result.kind is DimKind.AREA and result.area_m2 is not None:
            found.append(result.area_m2)
    remaining = list(found)
    matched = 0
    for target in areas:
        for i, value in enumerate(remaining):
            if abs(value - target) <= AREA_TOLERANCE * max(target, 1e-6):
                matched += 1
                remaining.pop(i)
                break
    return matched


def load_page(png: Path, pre: PreprocessConfig) -> np.ndarray:
    return preprocess(cv2.imread(str(png)), pre)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("DATASET_DIR", "/data/cubicasa5k")))
    parser.add_argument("--out", type=Path, default=Path("/eval/results/validate_ocr.json"))
    args = parser.parse_args()

    printed = json.loads(PRINTED_AREAS_PATH.read_text(encoding="utf-8"))["plans"]

    entries = [e for e in read_split(args.dataset, "test")
               if e.strip("/").startswith("high_quality_architectural")][: args.limit]

    default_pre = PreprocessConfig()
    report: dict[str, object] = {"plans": [], "ablations": {}}

    # ---- Part 1: dedup comparison, one OCR pass per plan ----
    print("=" * 78)
    print("DEDUP: confidence-only (before) vs vocabulary+orientation (after)")
    print("=" * 78)
    print(f"{'plan':>6} {'lbl':>4} {'svg':>4} {'prn':>4}  "
          f"{'labels b/a':>12}  {'areas b/a':>11}  note")

    raw_cache: dict[str, list[OcrToken]] = {}
    totals = {"lab_before": 0, "lab_after": 0, "ar_before": 0, "ar_after": 0,
              "lab_ref": 0, "ar_ref": 0, "ar_printed": 0,
              "ar_after_on_printing_plans": 0, "ar_spurious": 0}

    for entry in entries:
        directory = plan_dir(args.dataset, entry)
        plan_id = directory.name
        labels, areas = reference(directory / "model.svg")

        image = load_page(directory / "F1_scaled.png", default_pre)
        raw = run_ocr(image, min_confidence=0.5, dedup_iou=99.0)  # dedup disabled
        raw_cache[plan_id] = raw

        before = confidence_only_dedup(raw, 0.5)
        after = deduplicate(raw, 0.5, rule="tall_only")

        lb, la = count_labels(before, labels), count_labels(after, labels)
        ab, aa = count_areas(before, areas), count_areas(after, areas)

        totals["lab_before"] += lb
        totals["lab_after"] += la
        totals["ar_before"] += ab
        totals["ar_after"] += aa
        totals["lab_ref"] += len(labels)
        totals["ar_ref"] += len(areas)

        n_printed = printed.get(plan_id, {}).get("printed_area_count", 0)
        totals["ar_printed"] += n_printed
        if n_printed:
            totals["ar_after_on_printing_plans"] += aa
            note = ""
        else:
            # This plan prints no per-room areas, so anything matched here is a number that
            # coincidentally lands within tolerance of a polygon area - a false positive.
            totals["ar_spurious"] += aa
            note = "no areas printed" + (f"; {aa} spurious" if aa else "")

        print(f"{plan_id:>6} {len(labels):>4} {len(areas):>4} {n_printed:>4}  "
              f"{lb:>5} / {la:<4}  {ab:>4} / {aa:<4}  {note}")
        report["plans"].append({
            "plan": plan_id, "ref_labels": sorted(labels), "ref_areas": areas,
            "printed_area_count": n_printed,
            "labels_before": lb, "labels_after": la,
            "areas_before": ab, "areas_after": aa,
            "tokens_before": len(before), "tokens_after": len(after),
        })

    print("-" * 78)
    print(f"{'TOTAL':>6} {totals['lab_ref']:>4} {totals['ar_ref']:>4} "
          f"{totals['ar_printed']:>4}  "
          f"{totals['lab_before']:>5} / {totals['lab_after']:<4}  "
          f"{totals['ar_before']:>4} / {totals['ar_after']:<4}")
    print()
    print(f"  labels           {totals['lab_after']}/{totals['lab_ref']} "
          f"({100 * totals['lab_after'] / max(totals['lab_ref'], 1):.0f}%) "
          f"vs SVG label set")
    print(f"  areas  (svg)     {totals['ar_after']}/{totals['ar_ref']} "
          f"({100 * totals['ar_after'] / max(totals['ar_ref'], 1):.0f}%) "
          f"<- counts rooms whose area was never printed; misleading")
    print(f"  areas  (printed) {totals['ar_after_on_printing_plans']}/{totals['ar_printed']} "
          f"({100 * totals['ar_after_on_printing_plans'] / max(totals['ar_printed'], 1):.0f}%) "
          f"<- areas actually on the page, hand-counted")
    print(f"  spurious areas   {totals['ar_spurious']} "
          f"(matched on plans that print no areas at all)")
    report["dedup_totals"] = totals

    # ---- Part 2: setting ablations, each needs its own OCR pass ----
    variants = {
        "default (4 rot, 1536, no threshold)": dict(),
        "rotations 0 only": dict(orientations=[0]),
        "rotations 0,90": dict(orientations=[0, 90]),
        "det size 2400": dict(det_limit_side_len=2400),
        "adaptive threshold ON": dict(pre=PreprocessConfig(adaptive_threshold=True)),
    }

    print()
    print("=" * 78)
    print("ABLATIONS (new dedup throughout)")
    print("=" * 78)
    print(f"{'variant':<38} {'labels':>11} {'areas(printed)':>15} {'sec':>6}")

    for name, kwargs in variants.items():
        pre = kwargs.pop("pre", default_pre)
        lab = ar = ar_printed = 0
        t0 = time.time()
        for entry in entries:
            directory = plan_dir(args.dataset, entry)
            plan_id = directory.name
            labels, areas = reference(directory / "model.svg")
            image = load_page(directory / "F1_scaled.png", pre)
            tokens = run_ocr(image, min_confidence=0.5, dedup_iou=0.5, **kwargs)
            lab += count_labels(tokens, labels)
            matched = count_areas(tokens, areas)
            ar += matched
            if printed.get(plan_id, {}).get("printed_area_count", 0):
                ar_printed += matched
        secs = time.time() - t0
        print(f"{name:<38} {lab:>4}/{totals['lab_ref']:<6} "
              f"{ar_printed:>6}/{totals['ar_printed']:<8} {secs:>6.0f}")
        report["ablations"][name] = {
            "labels": lab, "areas_svg": ar, "areas_printed": ar_printed,
            "seconds": round(secs),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
