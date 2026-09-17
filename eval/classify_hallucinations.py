"""Classify every grounding failure in the 50-plan Tier 3 run, before and after the fix.

The Tier 3 cache rows record only a count. This replays grounding against the raw batch
output (fetched free by ``eval/retrieve_batch.py``) plus the cached OCR, and sorts every
failure the *original* checker raised into a cause:

``invented:*``
    A real fabrication. The number appears nowhere in the OCR text, or the cited token ids
    do not exist and the label is not on the page either.

``trusted_garbled_ocr:*``
    The model repeated a value that is present in a token, but in a token the parser treats
    as something else entirely (a door code, a wall run). Honest-but-wrong.

``scoring_artifact:*``
    The extraction was defensible and the checker was wrong. Overwhelmingly this is the
    model reading an area correctly out of a token OCR damaged - PaddleOCR renders the
    superscript in "m²" as LaTeX-like noise ("3,3 m^{2}$", "15.5 m^2}$") - which the regex
    parser cannot read. Holding the model to the parser's ceiling made it a hallucination to
    succeed where the parser failed. These are what the grounding fix removes.

No API calls. Free.

    python eval/classify_hallucinations.py --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "/app")

from app.pipeline.grounding import (
    _area_is_supported,
    _parsed_areas,
    _transcribed_numbers,
)
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef
from app.pipeline.parse_dims import DimKind, parse
from app.schemas import PlanExtraction

RAW_DIR = Path("/eval/cache/batch_raw")
OCR_DIR = Path("/eval/cache/ocr")

# vlm grounds softly by design and reports no hallucinations, so it has nothing to classify.
HARD_APPROACHES = ("ocr+llm", "hybrid")

# hybrid's prompt explicitly allows a room read off the image with no OCR token behind it,
# and ground_hybrid drops those, so they were never counted in the run either.
DROPS_MISSING_IDS = {"hybrid"}

SLUG_TO_APPROACH = {"ocr_llm": "ocr+llm", "vlm": "vlm", "hybrid": "hybrid"}


def load_tokens(plan_id: str) -> list[OcrToken] | None:
    path = OCR_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tokens" not in data:
        return None
    return [
        OcrToken(
            text=t["text"],
            confidence=t["confidence"],
            bbox=tuple(t["bbox"]),
            angle=t.get("angle", 0),
        )
        for t in data["tokens"]
    ]


def split_custom_id(stem: str) -> tuple[str, str]:
    """custom_id is safe_custom_id("<plan>|<approach>") = "<plan>_<slug>-<digest>"."""
    head = stem.rsplit("-", 1)[0]
    plan_id, _, slug = head.partition("_")
    return plan_id, SLUG_TO_APPROACH.get(slug, slug)


def classify_area(
    area: float, label: str, tokens: list[OcrToken], transcribed: list[float]
) -> tuple[str, str]:
    """Why did the original checker reject this area?"""
    if _area_is_supported(area, transcribed):
        # The number is in the OCR text; the parser just could not read that token.
        hits = [t.text for t in tokens if _token_contains(t.text, area, transcribed)]
        return (
            "scoring_artifact:area_in_garbled_token",
            f'"{label}" area {area:g} m2 is in token(s) {hits[:2]} that the parser '
            "cannot read as an area",
        )

    # Not transcribable. Is the number present at all, in something the parser calls a
    # door code or wall run? Then the model trusted a bad read rather than inventing.
    for token in tokens:
        result = parse(token.text)
        if result.kind is DimKind.DOOR_WINDOW_CODE and _digits_match(token.text, area):
            return (
                "trusted_garbled_ocr:door_code",
                f'"{label}" area {area:g} m2 comes from door/window code "{token.text}"',
            )
    return (
        "invented:area_not_on_page",
        f'"{label}" area {area:g} m2 appears nowhere in the OCR text',
    )


def _token_contains(text: str, area: float, transcribed: list[float]) -> bool:
    return _area_is_supported(area, _transcribed_numbers([_ref(text)])) and bool(transcribed)


def _ref(text: str) -> TokenRef:
    return TokenRef(id=0, text=text, confidence=1.0, bbox=(0, 0, 1, 1))


def _digits_match(text: str, area: float) -> bool:
    import re

    return any(
        abs(float(m.group(0).replace(",", ".")) - area) <= 0.02 * max(abs(area), 1e-6)
        for m in re.finditer(r"\d+(?:[.,]\d+)?", text)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("/eval/results/hallucinations.md"))
    args = parser.parse_args()

    counts: dict[str, Counter] = {a: Counter() for a in HARD_APPROACHES}
    examples: dict[str, dict[str, list[str]]] = {a: defaultdict(list) for a in HARD_APPROACHES}
    before: Counter = Counter()
    after: Counter = Counter()

    for raw_path in sorted(RAW_DIR.glob("*.json")):
        entry: dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))
        if "text" not in entry:
            continue
        plan_id, approach = split_custom_id(raw_path.stem)
        if approach not in HARD_APPROACHES:
            continue

        tokens = load_tokens(plan_id)
        if tokens is None:
            continue
        refs = TokenRef.from_tokens(tokens)
        try:
            extraction = PlanExtraction.model_validate(json.loads(entry["text"]))
        except Exception:  # noqa: BLE001 - a malformed body is worth counting, not crashing
            counts[approach]["invented:unparseable_response"] += 1
            continue

        parsed_areas = _parsed_areas(refs)
        transcribed = _transcribed_numbers(refs)
        valid_ids = {r.id for r in refs}

        for room in extraction.rooms:
            label = (room.label_raw or "").strip()

            # --- citation checks, exactly as the run applied them ---
            if approach not in DROPS_MISSING_IDS and not room.source_token_ids:
                before[approach] += 1
                after[approach] += 1
                on_page = [t.text for t in tokens if label and label.lower() in t.text.lower()]
                category = (
                    "scoring_artifact:uncited_but_on_page"
                    if on_page
                    else "invented:no_citation_no_evidence"
                )
                detail = (
                    f'room "{label}" cited no tokens'
                    + (f", but the label is on the page as {on_page[:2]}" if on_page else
                       " and the label is not in the OCR text")
                )
                counts[approach][category] += 1
                _record(examples, approach, category, plan_id, detail, args.top)
                continue

            if room.source_token_ids:
                unknown = [i for i in room.source_token_ids if i not in valid_ids]
                if unknown:
                    before[approach] += 1
                    after[approach] += 1
                    counts[approach]["invented:bad_citation"] += 1
                    _record(
                        examples, approach, "invented:bad_citation", plan_id,
                        f'room "{label}" cited non-existent id(s) {unknown} '
                        f"(max valid id {max(valid_ids) if valid_ids else -1})",
                        args.top,
                    )

            # --- area check ---
            if room.area_m2 is None:
                continue
            if _area_is_supported(room.area_m2, parsed_areas):
                continue
            # The original checker flagged this.
            before[approach] += 1
            category, detail = classify_area(room.area_m2, label, tokens, transcribed)
            if not category.startswith("scoring_artifact"):
                after[approach] += 1
            counts[approach][category] += 1
            _record(examples, approach, category, plan_id, detail, args.top)

    lines = ["# Hallucination classification - 50-plan Tier 3 run", ""]
    lines.append(
        "Replayed from the raw batch output. Retrieving a finished batch's results is free "
        "and they are retained for 29 days, so this cost nothing.\n"
    )
    lines.append(
        "`vlm` is absent because it grounds softly by design - the vision model may read an "
        "area OCR missed, so a disagreement with OCR would measure OCR - and it reported no "
        "hallucinations.\n"
    )
    lines.append("## Summary\n")
    lines.append("| Approach | Flagged before | Real after the fix | Removed as artifacts |")
    lines.append("| --- | --- | --- | --- |")
    for approach in HARD_APPROACHES:
        lines.append(
            f"| `{approach}` | {before[approach]} | {after[approach]} | "
            f"{before[approach] - after[approach]} |"
        )
    lines.append("")

    for approach in HARD_APPROACHES:
        total = sum(counts[approach].values())
        lines.append(f"## `{approach}` - {total} flagged by the original checker\n")
        lines.append("| Category | Count | Share |")
        lines.append("| --- | --- | --- |")
        for category, n in counts[approach].most_common():
            lines.append(f"| `{category}` | {n} | {100 * n / total:.0f}% |" if total else "")
        lines.append("")
        for category, _ in counts[approach].most_common():
            lines.append(f"**`{category}`**\n")
            for example in examples[approach][category]:
                lines.append(f"- {example}")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    for approach in HARD_APPROACHES:
        print(f"\n=== {approach}: {before[approach]} flagged before -> "
              f"{after[approach]} real ===")
        for category, n in counts[approach].most_common():
            print(f"  {category:<40} {n:4d}")
        for category, _ in counts[approach].most_common():
            for example in examples[approach][category][: args.top]:
                print(f"    [{category}] {example}")
    print(f"\nwrote {args.out}")
    return 0


def _record(
    examples: dict[str, dict[str, list[str]]],
    approach: str,
    category: str,
    plan_id: str,
    detail: str,
    top: int,
) -> None:
    if len(examples[approach][category]) < top:
        examples[approach][category].append(f"plan {plan_id}: {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
