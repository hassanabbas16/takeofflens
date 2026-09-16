"""Cost gate: run all three approaches on the 6 validation plans and report what they cost.

This is the gate before any larger run. It processes a small, fixed set of plans, reports
labels found, areas found, hallucinations, latency and **measured** cost per page for each
approach, and then stops. It never continues to a larger run on its own.

    python eval/tier3_cost_probe.py                 # the 6 validation plans
    python eval/tier3_cost_probe.py --approaches vlm

Requires OPENAI_API_KEY. Costs real money - roughly one text call and two vision calls per
plan. Results are cached per (plan, approach, model) so a re-run after a crash does not pay
twice; use --force to ignore the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "/app")

from app.config import get_settings  # noqa: E402
from app.llm import LlmClient, LlmError  # noqa: E402
from app.pipeline.extract import (  # noqa: E402
    ExtractionOutcome,
    extract_hybrid,
    extract_ocr_llm,
    extract_vlm,
)
from app.pipeline.ocr import OcrToken, run_ocr  # noqa: E402
from app.pipeline.preprocess import PreprocessConfig, preprocess  # noqa: E402
from app.pipeline.room_types import normalise_label  # noqa: E402
from app.pricing import get_pricing  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from svg_ground_truth import parse_model_svg, plan_dir, read_split  # noqa: E402

AREA_TOLERANCE = 0.05
PRINTED_AREAS_PATH = Path(__file__).parent / "ground_truth" / "printed_areas_6.json"
CACHE_DIR = Path("/eval/cache/llm")
OCR_CACHE_DIR = Path("/eval/cache/ocr")


def ocr_tokens_cached(png: Path, plan_id: str, force: bool = False) -> list[OcrToken]:
    """OCR once per plan and reuse, so a three-approach run does not OCR three times."""
    cache = OCR_CACHE_DIR / f"{plan_id}.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [
            OcrToken(text=t["text"], confidence=t["confidence"],
                     bbox=tuple(t["bbox"]), angle=t["angle"])
            for t in data["tokens"]
        ]
    image = preprocess(cv2.imread(str(png)), PreprocessConfig())
    tokens = run_ocr(image)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"tokens": [t.as_dict() for t in tokens]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return tokens


def reference(svg_path: Path) -> tuple[set[str], list[float]]:
    plan = parse_model_svg(svg_path)
    labels = {normalise_label(r.name) for r in plan.rooms if r.name}
    labels.discard("")
    labels.discard("UNDEFINED")
    return labels, [r.area_m2 for r in plan.rooms if r.area_m2 >= 1.0]


def score(outcome: ExtractionOutcome, labels: set[str], areas: list[float]) -> tuple[int, int]:
    found_labels = {normalise_label(r.label_raw) for r in outcome.rooms}
    matched_labels = sum(1 for label in labels if label in found_labels)

    remaining = [r.area_m2 for r in outcome.rooms if r.area_m2 is not None]
    matched_areas = 0
    for target in areas:
        for i, value in enumerate(remaining):
            if abs(value - target) <= AREA_TOLERANCE * max(target, 1e-6):
                matched_areas += 1
                remaining.pop(i)
                break
    return matched_labels, matched_areas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--approaches", nargs="*",
                        default=["ocr+llm", "vlm", "hybrid"])
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("DATASET_DIR", "/data/cubicasa5k")))
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/tier3_cost_probe.md"))
    parser.add_argument("--force", action="store_true", help="ignore cached LLM responses")
    args = parser.parse_args()

    settings = get_settings()
    pricing = get_pricing()

    try:
        client = LlmClient()
    except LlmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    printed = json.loads(PRINTED_AREAS_PATH.read_text(encoding="utf-8"))["plans"]
    entries = [e for e in read_split(args.dataset, "test")
               if e.strip("/").startswith("high_quality_architectural")][: args.limit]

    print(f"models: text={settings.openai_text_model} vision={settings.openai_vision_model}")
    print(f"pricing: {pricing.source} checked {pricing.checked} ({pricing.mode} mode)")
    for model in {settings.openai_text_model, settings.openai_vision_model}:
        if not pricing.knows(model):
            print(f"WARNING: no pricing for {model!r}; cost will be reported as unknown")
    print(f"plans: {len(entries)}   approaches: {args.approaches}\n")

    results: dict[str, list[dict]] = {a: [] for a in args.approaches}
    ref_totals = {"labels": 0, "areas_svg": 0, "areas_printed": 0}

    for entry in entries:
        directory = plan_dir(args.dataset, entry)
        plan_id = directory.name
        labels, areas = reference(directory / "model.svg")
        n_printed = printed.get(plan_id, {}).get("printed_area_count", 0)
        ref_totals["labels"] += len(labels)
        ref_totals["areas_svg"] += len(areas)
        ref_totals["areas_printed"] += n_printed

        tokens = ocr_tokens_cached(directory / "F1_scaled.png", plan_id)
        image = cv2.imread(str(directory / "F1_scaled.png"))
        print(f"{plan_id}: {len(tokens)} ocr tokens, {len(labels)} ref labels, "
              f"{n_printed} printed areas")

        for approach in args.approaches:
            cache = CACHE_DIR / approach.replace("+", "_") / f"{plan_id}.json"
            started = time.time()
            if approach == "ocr+llm":
                outcome = extract_ocr_llm(tokens, client)
            elif approach == "vlm":
                outcome = extract_vlm(image, client, tokens=tokens)
            elif approach == "hybrid":
                outcome = extract_hybrid(image, tokens, client)
            else:
                print(f"  unknown approach {approach!r}", file=sys.stderr)
                continue
            elapsed = time.time() - started

            if not outcome.ok:
                print(f"  {approach:<8} FAILED: {outcome.error}")
                results[approach].append({
                    "plan": plan_id, "ok": False, "error": outcome.error,
                    "latency_ms": outcome.usage.latency_ms,
                })
                continue

            matched_labels, matched_areas = score(outcome, labels, areas)
            cost = pricing.cost_usd(
                outcome.usage.model,
                outcome.usage.input_tokens,
                outcome.usage.output_tokens,
                outcome.usage.cached_input_tokens,
            )
            row = {
                "plan": plan_id, "ok": True,
                "rooms": len(outcome.rooms),
                "labels": matched_labels, "ref_labels": len(labels),
                "areas": matched_areas, "printed_areas": n_printed,
                "hallucinations": outcome.hallucinations,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "latency_ms": outcome.usage.latency_ms,
                "cost_usd": cost,
                "attempts": outcome.usage.attempts,
            }
            results[approach].append(row)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(row, indent=2), encoding="utf-8")

            cost_str = f"${cost:.5f}" if cost is not None else "unknown"
            print(f"  {approach:<8} rooms={len(outcome.rooms):<3} "
                  f"labels={matched_labels}/{len(labels)} areas={matched_areas}/{n_printed} "
                  f"halluc={outcome.hallucinations} {elapsed:.1f}s {cost_str}")

    # ---- summary ----
    lines: list[str] = []
    lines.append("# Tier 3 cost probe\n")
    lines.append(f"- Plans: {len(entries)} `high_quality_architectural` test plans")
    lines.append(f"- Text model: `{settings.openai_text_model}`")
    lines.append(f"- Vision model: `{settings.openai_vision_model}`")
    lines.append(f"- Pricing: {pricing.source}, checked {pricing.checked}, {pricing.mode} mode")
    lines.append(f"- Reference: {ref_totals['labels']} labels, "
                 f"{ref_totals['areas_printed']} areas actually printed "
                 f"({ref_totals['areas_svg']} rooms in the SVG annotation)\n")
    lines.append("| Approach | Labels | Areas (printed) | Hallucinations | "
                 "Mean latency | Mean cost/page | Total |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for approach in args.approaches:
        rows = [r for r in results[approach] if r.get("ok")]
        if not rows:
            lines.append(f"| {approach} | all calls failed | | | | | |")
            print(f"{approach}: all calls failed")
            continue
        labels = sum(r["labels"] for r in rows)
        areas = sum(r["areas"] for r in rows)
        halluc = sum(r["hallucinations"] for r in rows)
        latency = statistics.mean(r["latency_ms"] for r in rows)
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        mean_cost = statistics.mean(costs) if costs else None
        total_cost = sum(costs) if costs else None
        cost_cell = f"${mean_cost:.5f}" if mean_cost is not None else "unknown"
        total_cell = f"${total_cost:.4f}" if total_cost is not None else "unknown"
        lines.append(
            f"| {approach} | {labels}/{ref_totals['labels']} | "
            f"{areas}/{ref_totals['areas_printed']} | {halluc} | "
            f"{latency / 1000:.1f}s | {cost_cell} | {total_cell} |"
        )
        print(f"{approach:<8} labels={labels}/{ref_totals['labels']} "
              f"areas={areas}/{ref_totals['areas_printed']} halluc={halluc} "
              f"latency={latency / 1000:.1f}s cost/page={cost_cell} total={total_cell}")

    lines.append("\n## Extrapolation\n")
    lines.append("Cost of the full `high_quality_architectural` test split "
                 "(270 plans), at the measured per-page rate:\n")
    lines.append("| Approach | Projected cost |")
    lines.append("| --- | --- |")
    for approach in args.approaches:
        rows = [r for r in results[approach] if r.get("ok")]
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        if costs:
            lines.append(f"| {approach} | ${statistics.mean(costs) * 270:.2f} |")
        else:
            lines.append(f"| {approach} | unknown |")

    lines.append("\n**STOP.** This is the cost gate. The full run needs explicit approval.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"results": results, "reference": ref_totals}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    print("\nSTOP: cost gate. The full run needs explicit approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
