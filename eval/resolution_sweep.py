"""Find the smallest image the vision approaches can work from without losing recall.

Image tokens dominate the cost of ``vlm`` and ``hybrid``, and they scale with pixel count,
so halving the long edge cuts image tokens roughly fourfold. The question is where recall
starts to break: these plans carry small rotated labels, and past some size the model
simply cannot read them.

Two modes:

    --dry-run   arithmetic only. Projects image tokens and cost per resolution from the
                documented billing formula. No API calls, no cost.
    (default)   runs ``vlm`` and ``hybrid`` at each resolution over the validation plans
                and reports measured label recall, so the choice is made on evidence.

    python eval/resolution_sweep.py --dry-run
    python eval/resolution_sweep.py --resolutions 1536 1024 768 --max-spend 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/app")

from app.config import get_settings  # noqa: E402
from app.llm import LlmClient, LlmError  # noqa: E402
from app.pipeline.extract import encode_page_image, extract_hybrid, extract_vlm  # noqa: E402
from app.pipeline.ocr import OcrToken  # noqa: E402
from app.pipeline.room_types import label_key  # noqa: E402
from app.pricing import get_pricing  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from svg_ground_truth import parse_model_svg, plan_dir, read_split  # noqa: E402

OCR_CACHE = Path("/eval/cache/ocr")

# Anthropic bills an image at approximately (width x height) / 750 tokens. Used only for
# the dry-run projection; reported costs always come from logged token counts.
PIXELS_PER_IMAGE_TOKEN = 750

# Measured on the real run at 1536px: total input was ~3420 tokens, of which the image was
# the bulk; output averaged ~810 tokens.
PROMPT_OVERHEAD_TOKENS = 400
TYPICAL_OUTPUT_TOKENS = 810


def image_tokens(width: int, height: int) -> int:
    return int(width * height / PIXELS_PER_IMAGE_TOKEN)


def load_tokens(plan_id: str) -> list[OcrToken]:
    path = OCR_CACHE / f"{plan_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        OcrToken(text=t["text"], confidence=t["confidence"],
                 bbox=tuple(t["bbox"]), angle=t["angle"])
        for t in data["tokens"]
    ]


def _reference_labels(svg_path: Path) -> set[str]:
    plan = parse_model_svg(svg_path)
    labels = {label_key(r.name) for r in plan.rooms if r.name}
    return labels - {"", "UNDEFINED"}


def _dry_run(args, settings, pricing, entries, model) -> int:
    print("DRY RUN - arithmetic only, no API calls\n")
    print(f"model {model}, {len(entries)} plans")
    print(f"{'long edge':>10} {'mean img tokens':>16} {'est $/page':>12} {'vs 1536':>9}")
    baseline = None
    for res in sorted(args.resolutions, reverse=True):
        total = 0
        for entry in entries:
            directory = plan_dir(args.dataset, entry)
            image = cv2.imread(str(directory / "F1_scaled.png"))
            _, width, height = encode_page_image(image, res, settings.vlm_jpeg_quality)
            total += image_tokens(width, height)
        mean_tokens = total / len(entries)
        cost = pricing.cost_usd(
            model, int(mean_tokens + PROMPT_OVERHEAD_TOKENS), TYPICAL_OUTPUT_TOKENS
        ) or 0.0
        if baseline is None:
            baseline = cost
        share = 100 * cost / baseline if baseline else 100
        print(f"{res:>10} {mean_tokens:>16,.0f} ${cost:>11.5f} {share:>8.0f}%")
    print("\nThis projects COST only. Whether recall survives at each size needs real "
          "calls - run without --dry-run.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="*", default=[1536, 1024, 768, 512])
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true",
                        help="project tokens and cost arithmetically; make no API calls")
    parser.add_argument("--max-spend", type=float, default=0.30)
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/resolution_sweep.md"))
    args = parser.parse_args()

    settings = get_settings()
    pricing = get_pricing()
    model = settings.anthropic_vision_model

    entries = [e for e in read_split(args.dataset, "test")
               if e.strip("/").startswith("high_quality_architectural")][: args.limit]

    if args.dry_run:
        return _dry_run(args, settings, pricing, entries, model)

    try:
        client = LlmClient()
    except LlmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"api key: {settings.api_key_source}")
    print(f"model {model}, {len(entries)} plans, cap ${args.max_spend:.2f}\n")

    spend = 0.0
    rows: list[dict] = []
    for res in sorted(args.resolutions, reverse=True):
        for approach in ("vlm", "hybrid"):
            labels_found = ref_total = 0
            approach_cost = 0.0
            for entry in entries:
                if spend + 0.010 > args.max_spend:
                    print(f"STOPPING at spend cap (${spend:.4f})")
                    _write(args.out, rows, model, spend)
                    return 0
                directory = plan_dir(args.dataset, entry)
                reference = _reference_labels(directory / "model.svg")
                tokens = load_tokens(directory.name)
                image = cv2.imread(str(directory / "F1_scaled.png"))

                # The encoder reads this from settings, so the sweep sets it per pass.
                object.__setattr__(settings, "vlm_max_image_px", res)

                if approach == "vlm":
                    outcome = extract_vlm(image, client, tokens=tokens)
                else:
                    outcome = extract_hybrid(image, tokens, client)

                if not outcome.ok:
                    print(f"  {res} {approach} {directory.name}: FAILED {outcome.error}")
                    continue
                found = {label_key(r.label_raw) for r in outcome.rooms}
                labels_found += sum(1 for label in reference if label in found)
                ref_total += len(reference)
                cost = pricing.cost_usd(
                    outcome.usage.model,
                    outcome.usage.input_tokens,
                    outcome.usage.output_tokens,
                    outcome.usage.cache_read_tokens,
                    outcome.usage.cache_write_tokens,
                ) or 0.0
                approach_cost += cost
                spend += cost
            rows.append({"resolution": res, "approach": approach,
                         "labels": labels_found, "ref": ref_total, "cost": approach_cost})
            print(f"  {res:>5} {approach:<7} labels={labels_found}/{ref_total} "
                  f"${approach_cost:.4f}  (running ${spend:.4f})")

    _write(args.out, rows, model, spend)
    return 0


def _write(out: Path, rows: list[dict], model: str, spend: float) -> None:
    lines = [
        "# Image resolution sweep\n",
        f"- Model: `{model}`",
        f"- Measured spend: ${spend:.4f}\n",
        "| Long edge | Approach | Label recall | Cost |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {row['approach']} | "
            f"{row['labels']}/{row['ref']} | ${row['cost']:.4f} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
