"""Measure what the OCR area-unit repair buys, on the 50-plan Tier 3 sample.

Compares three arms over the cached OCR, with no API calls and no cost:

``off``
    the parser as it was before ``normalise_area_unit`` existed.

``on``
    the shipped repair: strip LaTeX ``$`` delimiters, and collapse an ``m`` followed by
    superscript debris (``^``, ``{``, ``}``, an optional ``2``) into ``m2``.

``aggressive``
    ``on`` plus treating a bare trailing ``m`` after a number as ``m2`` ("17.6m" -> 17.6 m2).
    Measured to justify leaving it out of the shipped repair rather than to ship it: a bare
    ``m`` is a plausible length, so this arm trades a guess for recall.

Reported at two levels, because they answer different questions:

- **tokens**: how many OCR tokens the parser can read as an area at all. Pure parser reach.
- **rules rooms/areas**: what the free `rules` approach actually outputs end to end, which
  is what the results table reports. The corroboration and pairing rules sit between the
  two, so the second number is always the smaller.

    python eval/normalisation_sweep.py

`rules` is deterministic and free, so both arms are recomputed from scratch here. The LLM
approaches are **not** re-run - their cached outputs were produced against pre-repair
prompts and re-running them would cost money.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline import parse_dims
from app.pipeline.extract import extract_rules
from app.pipeline.ocr import OcrToken
from app.pipeline.parse_dims import DimKind

OCR_CACHE_DIR = Path("/eval/cache/ocr")
RESULTS_JSON = Path("/eval/results/tier3_sample50.json")
PRINTED = Path("/eval/ground_truth/printed_areas_6.json")

_TRAILING_M = re.compile(r"(?i)(\d)\s*m\s*$")
_REAL = parse_dims.normalise_area_unit


def arm_off(text: str) -> str:
    return text


def arm_on(text: str) -> str:
    return _REAL(text)


def arm_aggressive(text: str) -> str:
    return _TRAILING_M.sub(r"\1 m2", _REAL(text))


ARMS = {"off": arm_off, "on": arm_on, "aggressive": arm_aggressive}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/normalisation_sweep.md"))
    args = parser.parse_args()

    plan_ids = sample_plan_ids()
    printed = json.loads(PRINTED.read_text(encoding="utf-8"))["plans"]

    stats = {
        name: {"tokens": 0, "rooms": 0, "areas": 0, "plans_changed": set(),
               "fab": 0, "fab_plans": set()}
        for name in ARMS
    }
    total_tokens = 0
    examples: list[str] = []

    original = parse_dims.normalise_area_unit
    try:
        for plan_id in plan_ids:
            tokens = load_tokens(plan_id)
            if tokens is None:
                continue
            total_tokens += len(tokens)
            n_printed = printed.get(plan_id, {}).get("printed_area_count")

            for name, fn in ARMS.items():
                # Swap the module-level repair so both the parser and everything downstream
                # of it (pairing, extract_rules) see this arm consistently.
                parse_dims.normalise_area_unit = fn
                readable = 0
                for token in tokens:
                    result = parse_dims.parse(token.text)
                    if (
                        result.kind is DimKind.AREA
                        and result.area_m2 is not None
                        and result.plausible
                        and not result.area_needs_corroboration
                    ):
                        readable += 1
                        if (
                            name == "on"
                            and fn(token.text) != token.text
                            and len(examples) < 10
                        ):
                            examples.append(
                                f'`{token.text}` -> `{fn(token.text)}` = {result.area_m2:g} m2'
                            )
                outcome = extract_rules(tokens)
                areas = [r for r in outcome.rooms if r.area_m2 is not None]
                stats[name]["tokens"] += readable
                stats[name]["rooms"] += len(outcome.rooms)
                stats[name]["areas"] += len(areas)
                # Plans the 6-plan hand-verified set says print no areas at all: anything
                # reported there is a fabrication, and a recall change must not buy one.
                if n_printed == 0 and areas:
                    stats[name]["fab"] += len(areas)
                    stats[name]["fab_plans"].add(plan_id)

            parse_dims.normalise_area_unit = arm_off
            base = len([r for r in extract_rules(tokens).rooms if r.area_m2 is not None])
            parse_dims.normalise_area_unit = arm_on
            now = len([r for r in extract_rules(tokens).rooms if r.area_m2 is not None])
            if now != base:
                stats["on"]["plans_changed"].add(plan_id)
    finally:
        parse_dims.normalise_area_unit = original

    covered = [p for p in plan_ids if p in printed]
    lines = [
        "# OCR area-unit repair: what it buys",
        "",
        f"- {len(plan_ids)} `high_quality_architectural` test plans (the Tier 3 sample), "
        f"{total_tokens} OCR tokens",
        "- Free: `rules` is deterministic and recomputed from the OCR cache in every arm.",
        "",
        "> **The `rules` row is the only one that moves.** The `ocr+llm`, `vlm` and `hybrid`",
        "> numbers in `tier3_sample50.md` were produced against **pre-repair** prompts and",
        "> are not re-run here - that would cost money. They are therefore not comparable to",
        "> a post-repair `rules` on the pairing that fed them, and the results table says so",
        "> next to the figures.",
        "",
        "| Arm | Area-readable tokens | `rules` rooms | `rules` areas | Fabricated areas |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("off", "on", "aggressive"):
        s = stats[name]
        lines.append(
            f"| `{name}` | {s['tokens']} | {s['rooms']} | {s['areas']} | {s['fab']} |"
        )
    off, on, agg = stats["off"], stats["on"], stats["aggressive"]
    lines += [
        "",
        f"Repair moves area-readable tokens **{off['tokens']} -> {on['tokens']}** "
        f"(+{on['tokens'] - off['tokens']}) and `rules` areas "
        f"**{off['areas']} -> {on['areas']}** (+{on['areas'] - off['areas']}), "
        f"changing the output on {len(on['plans_changed'])} of {len(plan_ids)} plans.",
        "",
        f"The fabrication column is over the {len(covered)} sampled plans that have "
        "hand-verified printed-area ground truth, counting areas reported on a plan the "
        "human recorded as printing none. It must not rise.",
        "",
        "## Why the aggressive arm is not shipped",
        "",
        f"It adds {agg['areas'] - on['areas']} further `rules` areas over `on`, by assuming "
        "a bare trailing `m` means `m2`. That is a guess about a token that could legitimately "
        "be a length, not a repair of a known recogniser defect, and it is the kind of "
        "assumption this project reports rather than makes.",
        "",
        "## Repairs that fired",
        "",
    ]
    lines += [f"- {e}" for e in examples]
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(plan_ids)} plans, {total_tokens} tokens\n")
    print(f"{'arm':<12} {'readable tokens':>16} {'rules rooms':>12} {'rules areas':>12} "
          f"{'fabricated':>11}")
    for name in ("off", "on", "aggressive"):
        s = stats[name]
        print(f"{name:<12} {s['tokens']:>16} {s['rooms']:>12} {s['areas']:>12} "
              f"{s['fab']:>11}")
    print(f"\nplans whose rules output changed: {len(on['plans_changed'])}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
