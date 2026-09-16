"""Cost gate: run all three approaches on the 6 validation plans and report what they cost.

This is the gate before any larger run. It processes a small, fixed set of plans, reports
labels found, areas found, hallucinations, latency and **measured** cost per page for each
approach, and then stops. It never continues to a larger run on its own.

    python eval/tier3_cost_probe.py                 # the 6 validation plans
    python eval/tier3_cost_probe.py --approaches vlm

Requires ANTHROPIC_API_KEY. Costs real money - roughly one text call and two vision calls per
plan. Results are cached per (plan, approach, model) so a re-run after a crash does not pay
twice; use --force to ignore the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "/app")

from app.config import get_settings  # noqa: E402
from app.llm import LlmClient, LlmError  # noqa: E402
from app.batch import BatchItem, submit_and_wait  # noqa: E402
from app.pipeline.extract import (  # noqa: E402
    ExtractionOutcome,
    build_hybrid_prompt,
    build_ocr_llm_prompt,
    build_vlm_prompt,
    encode_page_image,
    extract_hybrid,
    extract_ocr_llm,
    extract_rules,
    extract_vlm,
    ground_hybrid,
    ground_ocr_llm,
    ground_vlm,
    image_content,
    system_prompt,
    text_content,
)
from app.pipeline.pairing import TokenRef  # noqa: E402
from app.schemas import Source  # noqa: E402
from app.pipeline.ocr import OcrToken, run_ocr  # noqa: E402
from app.pipeline.preprocess import PreprocessConfig, preprocess  # noqa: E402
from app.pipeline.room_types import label_key  # noqa: E402
from app.pricing import get_pricing  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from svg_ground_truth import parse_model_svg, plan_dir, read_split  # noqa: E402

AREA_TOLERANCE = 0.05

# Fixed seed for the Tier 3 random sample, so the plan set is reproducible and auditable.
# Recorded here rather than passed ad hoc: a sample nobody can regenerate is not a sample.
SAMPLE_SEED = 20260917

# Used by the spend cap before any page has been measured in this run.
DEFAULT_PAGE_ESTIMATE = 0.010
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
    labels = {label_key(r.name) for r in plan.rooms if r.name}
    labels.discard("")
    labels.discard("UNDEFINED")
    return labels, [r.area_m2 for r in plan.rooms if r.area_m2 >= 1.0]


def score(outcome: ExtractionOutcome, labels: set[str], areas: list[float]) -> tuple[int, int]:
    found_labels = {label_key(r.label_raw) for r in outcome.rooms}
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


def build_batch_item(
    approach: str, plan_id: str, tokens, image, settings
) -> BatchItem:
    """One batch request for (plan, approach), using the same prompts as the sync path."""
    refs = TokenRef.from_tokens(tokens)
    if approach == "ocr+llm":
        model = settings.anthropic_text_model
        messages = [{"role": "user", "content": build_ocr_llm_prompt(refs)}]
    else:
        model = settings.anthropic_vision_model
        encoded, width, height = encode_page_image(
            image, settings.vlm_max_image_px, settings.vlm_jpeg_quality
        )
        prompt = (
            build_vlm_prompt(width, height)
            if approach == "vlm"
            else build_hybrid_prompt(refs, width, height)
        )
        messages = [{
            "role": "user",
            "content": [text_content(prompt), image_content(encoded)],
        }]
    return BatchItem(
        custom_id=f"{plan_id}|{approach}",
        model=model,
        system=system_prompt(),
        messages=messages,
        max_tokens=settings.anthropic_max_tokens,
    )


def ground_for(approach: str, parsed, refs):
    if approach == "ocr+llm":
        return ground_ocr_llm(parsed, refs)
    if approach == "vlm":
        return ground_vlm(parsed, refs)
    return ground_hybrid(parsed, refs)


_SOURCE_FOR = {
    "ocr+llm": Source.OCR_LLM,
    "vlm": Source.VLM,
    "hybrid": Source.HYBRID,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6,
                        help="number of plans (the 6 validation plans by default)")
    parser.add_argument("--sample", type=int, default=None,
                        help="instead of the first N, take a random sample of this size")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED,
                        help="RNG seed for --sample (default: %(default)s, documented)")
    parser.add_argument("--approaches", nargs="*",
                        default=["rules", "ocr+llm", "vlm", "hybrid"])
    parser.add_argument("--max-spend", type=float, default=None,
                        help=("hard cap in USD. Stops before a call that would take the "
                              "running total past this, based on logged costs so far."))
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("DATASET_DIR", "/data/cubicasa5k")))
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/tier3_cost_probe.md"))
    parser.add_argument("--force", action="store_true", help="ignore cached LLM responses")
    parser.add_argument("--batch", action="store_true",
                        help="use the Message Batches API (50%% of standard rates)")
    args = parser.parse_args()

    settings = get_settings()
    pricing = get_pricing()

    # The client is built on first actual need, not up front: a run served entirely from
    # cache, or one using only the free rules approach, must not require a key at all.
    client_box: dict[str, LlmClient] = {}

    def get_client() -> LlmClient | None:
        if "client" not in client_box:
            try:
                client_box["client"] = LlmClient()
            except LlmError as exc:
                print(f"error: {exc}", file=sys.stderr)
                client_box["client"] = None  # type: ignore[assignment]
        return client_box["client"]

    printed = json.loads(PRINTED_AREAS_PATH.read_text(encoding="utf-8"))["plans"]
    all_arch = [e for e in read_split(args.dataset, "test")
                if e.strip("/").startswith("high_quality_architectural")]
    if args.sample:
        rng = random.Random(args.seed)
        entries = sorted(rng.sample(all_arch, min(args.sample, len(all_arch))))
        print(f"sample: {len(entries)} of {len(all_arch)} plans, seed {args.seed}")
    else:
        entries = all_arch[: args.limit]

    print(f"api key: {settings.api_key_source}")
    print(f"models: text={settings.anthropic_text_model} "
          f"vision={settings.anthropic_vision_model}")
    print(f"pricing: {pricing.source} checked {pricing.checked} ({pricing.mode} mode)")
    for model in {settings.anthropic_text_model, settings.anthropic_vision_model}:
        if not pricing.knows(model):
            print(f"WARNING: no pricing for {model!r}; cost will be reported as unknown")
    print(f"plans: {len(entries)}   approaches: {args.approaches}\n")

    results: dict[str, list[dict]] = {a: [] for a in args.approaches}
    ref_totals = {"labels": 0, "areas_svg": 0, "areas_printed": 0}
    spend = {"total": 0.0, "worst_page": 0.0}
    stop_reason: str | None = None

    # ---- batch pass -------------------------------------------------------------------
    # Submits every uncached (plan, approach) in one batch at half price, then fills the
    # cache so the loop below reads results rather than paying again. `rules` never goes to
    # the API.
    if args.batch:
        batch_approaches = [a for a in args.approaches if a != "rules"]
        pending: list = []
        for entry in entries:
            directory = plan_dir(args.dataset, entry)
            plan_id = directory.name
            tokens = ocr_tokens_cached(directory / "F1_scaled.png", plan_id)
            image = None
            for approach in batch_approaches:
                cache = CACHE_DIR / approach.replace("+", "_") / f"{plan_id}.json"
                if cache.exists() and not args.force:
                    continue
                if image is None and approach != "ocr+llm":
                    image = cv2.imread(str(directory / "F1_scaled.png"))
                pending.append((
                    plan_id, approach, tokens,
                    build_batch_item(approach, plan_id, tokens, image, settings),
                ))

        if pending:
            # Project the bill before submitting. A batch is paid for as a whole, so the
            # per-call spend cap cannot stop it midway - the check has to happen here.
            per_page = {"ocr+llm": 0.00618, "vlm": 0.00749, "hybrid": 0.00882}
            projected = sum(
                per_page.get(approach, 0.010) * pricing.batch_multiplier
                for _, approach, _, _ in pending
            )
            print(f"\nbatch: {len(pending)} requests, projected "
                  f"${projected:.4f} at {pricing.batch_multiplier:.0%} of standard rates")
            if args.max_spend is not None and projected > args.max_spend:
                print(f"REFUSING to submit: projected ${projected:.4f} exceeds the "
                      f"${args.max_spend:.2f} cap", file=sys.stderr)
                return 2

            client = get_client()
            if client is None:
                return 2

            print("submitting batch and waiting (this takes minutes, not seconds)...")
            outcomes = submit_and_wait(
                client.raw,
                [item for _, _, _, item in pending],
                on_progress=lambda b: print(f"  batch {b.id}: {b.processing_status}"),
            )

            for plan_id, approach, tokens, item in pending:
                outcome = outcomes.get(item.custom_id)
                directory = plan_dir(args.dataset, f"/high_quality_architectural/{plan_id}/")
                labels, areas = reference(directory / "model.svg")
                n_printed = printed.get(plan_id, {}).get("printed_area_count", 0)
                cache = CACHE_DIR / approach.replace("+", "_") / f"{plan_id}.json"
                cache.parent.mkdir(parents=True, exist_ok=True)

                if outcome is None or not outcome.ok:
                    error = outcome.error if outcome else "missing from batch results"
                    cache.write_text(json.dumps(
                        {"plan": plan_id, "ok": False, "error": error, "latency_ms": 0,
                         "batch": True}, indent=2), encoding="utf-8")
                    print(f"  {plan_id} {approach}: FAILED {error}")
                    continue

                refs = TokenRef.from_tokens(tokens)
                grounded = ground_for(approach, outcome.parsed, refs)
                scored = ExtractionOutcome(
                    _SOURCE_FOR[approach], grounded, outcome.usage, refs
                )
                matched_labels, matched_areas = score(scored, labels, areas)
                cost = pricing.cost_usd(
                    outcome.usage.model, outcome.usage.input_tokens,
                    outcome.usage.output_tokens, outcome.usage.cache_read_tokens,
                    outcome.usage.cache_write_tokens, batch=True,
                )
                row = {
                    "plan": plan_id, "ok": True, "batch": True,
                    "rooms": len(scored.rooms),
                    "labels": matched_labels, "ref_labels": len(labels),
                    "areas": matched_areas, "printed_areas": n_printed,
                    "areas_on_printing_plans": matched_areas if n_printed else 0,
                    "spurious_areas": 0 if n_printed else matched_areas,
                    "hallucinations": scored.hallucinations,
                    "input_tokens": outcome.usage.input_tokens,
                    "output_tokens": outcome.usage.output_tokens,
                    "latency_ms": 0,  # asynchronous; per-request latency is not meaningful
                    "cost_usd": cost,
                    "attempts": 1,
                }
                cache.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print("batch complete; results cached\n")

    for entry in entries:
        if stop_reason:
            break
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
            if stop_reason:
                break
            cache = CACHE_DIR / approach.replace("+", "_") / f"{plan_id}.json"
            if cache.exists() and not args.force:
                row = json.loads(cache.read_text(encoding="utf-8"))
                # Backfill fields added after a cache entry was written, so an older cache
                # does not silently report zeros.
                if row.get("ok") and "areas_on_printing_plans" not in row:
                    matched = row.get("areas", 0)
                    printed_here = row.get("printed_areas", 0)
                    row["areas_on_printing_plans"] = matched if printed_here else 0
                    row["spurious_areas"] = 0 if printed_here else matched
                results[approach].append(row)
                cost_str = (
                    f"${row['cost_usd']:.5f}" if row.get("cost_usd") is not None else "unknown"
                )
                print(f"  {approach:<8} (cached) labels={row.get('labels')}/{len(labels)} "
                      f"areas={row.get('areas')}/{n_printed} "
                      f"halluc={row.get('hallucinations')} {cost_str}")
                continue
            # Spend cap: refuse to start a call that could take the total past the cap,
            # estimating from the most expensive page seen so far. Checked BEFORE the call,
            # because afterwards the money is already spent.
            if args.max_spend is not None and approach != "rules":
                estimate = max(spend["worst_page"], DEFAULT_PAGE_ESTIMATE)
                if spend["total"] + estimate > args.max_spend:
                    stop_reason = (
                        f"spend cap reached: ${spend['total']:.4f} spent, the next call "
                        f"could cost up to ${estimate:.4f}, cap is ${args.max_spend:.2f}"
                    )
                    print(f"  STOPPING - {stop_reason}")
                    break

            started = time.time()
            if approach == "rules":
                outcome = extract_rules(tokens)
            else:
                client = get_client()
                if client is None:
                    stop_reason = "no API key available for model-backed approaches"
                    print(f"  STOPPING - {stop_reason}")
                    break
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
            # The rules approach makes no call, so its cost is exactly zero - not
            # unknown. Reporting it as unknown would invite a reader to assume it
            # might be expensive.
            cost = 0.0 if approach == "rules" else pricing.cost_usd(
                outcome.usage.model,
                outcome.usage.input_tokens,
                outcome.usage.output_tokens,
                outcome.usage.cache_read_tokens,
                outcome.usage.cache_write_tokens,
            )
            row = {
                "plan": plan_id, "ok": True,
                "rooms": len(outcome.rooms),
                "labels": matched_labels, "ref_labels": len(labels),
                "areas": matched_areas, "printed_areas": n_printed,
                # An area "matched" on a plan that prints none is a number that happened to
                # land within tolerance of a polygon area - a false positive, not a hit.
                "areas_on_printing_plans": matched_areas if n_printed else 0,
                "spurious_areas": 0 if n_printed else matched_areas,
                "hallucinations": outcome.hallucinations,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "latency_ms": outcome.usage.latency_ms,
                "cost_usd": cost,
                "attempts": outcome.usage.attempts,
            }
            results[approach].append(row)
            if cost:
                spend["total"] += cost
                spend["worst_page"] = max(spend["worst_page"], cost)
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
    lines.append(f"- Text model: `{settings.anthropic_text_model}`")
    lines.append(f"- Vision model: `{settings.anthropic_vision_model}`")
    lines.append(f"- Pricing: {pricing.source}, checked {pricing.checked}, {pricing.mode} mode")
    lines.append(f"- Reference: {ref_totals['labels']} labels, "
                 f"{ref_totals['areas_printed']} areas actually printed "
                 f"({ref_totals['areas_svg']} rooms in the SVG annotation)\n")
    lines.append("| Approach | Labels | Areas (printed) | Spurious areas | Hallucinations | "
                 "Mean latency | Mean cost/page | Total |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for approach in args.approaches:
        rows = [r for r in results[approach] if r.get("ok")]
        if not rows:
            lines.append(f"| {approach} | all calls failed | | | | | | |")
            print(f"{approach}: all calls failed")
            continue
        labels = sum(r["labels"] for r in rows)
        areas = sum(r.get("areas_on_printing_plans", 0) for r in rows)
        spurious = sum(r.get("spurious_areas", 0) for r in rows)
        halluc = sum(r["hallucinations"] for r in rows)
        latency = statistics.mean(r["latency_ms"] for r in rows)
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        mean_cost = statistics.mean(costs) if costs else None
        total_cost = sum(costs) if costs else None
        cost_cell = f"${mean_cost:.5f}" if mean_cost is not None else "unknown"
        total_cell = f"${total_cost:.4f}" if total_cost is not None else "unknown"
        lines.append(
            f"| {approach} | {labels}/{ref_totals['labels']} | "
            f"{areas}/{ref_totals['areas_printed']} | {spurious} | {halluc} | "
            f"{latency / 1000:.1f}s | {cost_cell} | {total_cell} |"
        )
        print(f"{approach:<8} labels={labels}/{ref_totals['labels']} "
              f"areas={areas}/{ref_totals['areas_printed']} spurious={spurious} "
              f"halluc={halluc} latency={latency / 1000:.1f}s cost/page={cost_cell} "
              f"total={total_cell}")

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

    if stop_reason:
        lines.append(f"\n> **RUN STOPPED EARLY** - {stop_reason}. The table above covers "
                     "only the plans completed before the cap.\n")
    cap_note = f" (cap ${args.max_spend:.2f})" if args.max_spend else ""
    lines.append(f"\nMeasured spend this run: **${spend['total']:.4f}**{cap_note}\n")
    lines.append("\n**STOP.** This is the cost gate. The full run needs explicit approval.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"results": results, "reference": ref_totals}, indent=2), encoding="utf-8"
    )
    cap_note = f" (cap ${args.max_spend:.2f})" if args.max_spend else ""
    print(f"\nmeasured spend this run: ${spend['total']:.4f}{cap_note}")
    if stop_reason:
        print(f"RUN STOPPED EARLY: {stop_reason}")
    print(f"wrote {args.out}")
    print("\nSTOP: cost gate. The full run needs explicit approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
