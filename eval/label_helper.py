"""Optional tool: read the areas a drawing prints, one room at a time.

    docker compose run --rm --no-deps api python /eval/label_helper.py 1191 2207
    docker compose run --rm --no-deps api python /eval/label_helper.py --review 1041

**Nothing in the evaluation depends on this.** Area accuracy is measured against the
``model.svg`` polygons over all 50 sampled plans (``eval/tier2_area_accuracy.py``), which is
free and needs no human. This script is kept because reading a handful of plans by hand is
the fastest way to understand a disagreement between an approach and the polygons - it is a
debugging aid, not a source of metrics.

Plan ids are passed on the command line. Output lands in
``eval/ground_truth/manual/<id>.json`` and is not read by any report.

What it records
---------------
The **areas printed on the drawing**, with the label each is printed under. Not the SVG's
polygon areas: ``model.svg`` is CubiCasa's annotation of the page, its own dimension labels
are ``display:none`` and never rendered, and its polygon area follows the inner wall face
rather than the figure the agent printed. The SVG is a prefill and a checklist here, never
an oracle.

Deliberately *not* drawn on the page image: the SVG room polygons. The SVG coordinate space
does not map to ``F1_scaled.png`` by any constant - measured on three plans the per-axis
ratios are 0.909/1.002, 0.949/1.065 and 1.025/1.054 - so drawing them would put boxes in
confidently wrong places. OCR token boxes *are* in image pixel space, so those are drawn
with their ids, and the CLI refers to rooms by the token ids you can see on the page.

The flow
--------
For each plan it writes ``eval/labelling/<id>.png`` with the OCR boxes and ids drawn on, then
walks the SVG's room list. For each room it shows the machine's best guess at the printed
area - taken from the same bbox pairing the pipeline uses - and the OCR tokens near it, and
asks you to confirm, correct, or say the drawing prints no area there. Then it offers any
remaining parseable area on the page that no room claimed, in case the SVG missed a room.

Every answer is written to ``eval/ground_truth/manual/<id>.json`` immediately, so the work is
resumable at any point: re-running continues at the first undecided room.

Each field records how it was decided, so an area a human read off the drawing is
distinguishable from one accepted unchanged from the machine's suggestion.

No API calls. Free.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.ocr import OcrToken, draw_tokens
from app.pipeline.pairing import weighted_distance
from app.pipeline.parse_dims import (
    MAX_PLAUSIBLE_AREA_M2 as MAX_AREA,
)
from app.pipeline.parse_dims import (
    MIN_PLAUSIBLE_AREA_M2 as MIN_AREA,
)
from app.pipeline.parse_dims import DimKind, parse
from app.pipeline.room_types import normalise_label

from svg_ground_truth import parse_model_svg, plan_dir

OUT_DIR_JSON = Path("/eval/ground_truth/manual")
LABELLING_DIR = Path("/eval/labelling")
OCR_CACHE_DIR = Path("/eval/cache/ocr")

# Mirrors the pairing stage: an area sits within a few label-heights of its label.
MAX_LABEL_HEIGHTS = 6.0

# Parser verdicts that positively mean "not a room area".
_NOT_AN_AREA = frozenset({"elevation", "scale", "apartment_summary", "bare_integer"})
_DECIMAL = re.compile(r"\d+[.,]\d+")
_INT_WITH_UNIT = re.compile(r"(\d+)\s*[mM]\s*[\^{\s]{0,3}[2²]")

HELP = """
  Enter / y   accept the suggested area as printed
  <number>    the drawing prints this instead (e.g. 11.7 or 11,7)
  n           the drawing prints NO area for this room  (a real answer, not a skip)
  l           correct the label
  s           skip - decide later, stays undecided
  b           back one room
  ?           show this help again
  q           save and quit
"""


def load_tokens(plan_id: str) -> list[OcrToken]:
    path = OCR_CACHE_DIR / f"{plan_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        OcrToken(
            text=t["text"], confidence=t["confidence"],
            bbox=tuple(t["bbox"]), angle=t.get("angle", 0),
        )
        for t in data["tokens"]
    ]


def render(plan_id: str, png: Path, tokens: list[OcrToken]) -> Path:
    """Page image with OCR boxes and ids drawn, for reference while labelling."""
    LABELLING_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELLING_DIR / f"{plan_id}.png"
    if out.exists():
        return out
    image = cv2.imread(str(png))
    canvas = draw_tokens(image, tokens)
    for i, token in enumerate(tokens):
        x1, y1, _, _ = token.bbox
        cv2.putText(canvas, str(i), (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)
    return out


def label_token_indices(tokens: list[OcrToken], label: str) -> list[int]:
    """Tokens that *are* this label, not merely contain its letters.

    Substring matching is useless here: the one-character kitchen label "K" is inside RUOK,
    JKK and SK. Matching is on the normalised label being equal to the whole token, or to
    one whitespace-separated word of it (so "MH 11.7" still matches "MH").
    """
    key = normalise_label(label)
    if not key:
        return []
    hits = []
    for i, token in enumerate(tokens):
        norm = normalise_label(token.text)
        if norm == key or key in [normalise_label(w) for w in token.text.split()]:
            hits.append(i)
    return hits


def readable_numbers(text: str) -> list[float]:
    """Every number a human could read as an area in this token, parser or not.

    A parsed area first; otherwise any decimal, or an integer carrying a damaged area unit.
    OCR garbles area text constantly on these plans ("9,5m^2}$", "OH. 16.0 n?"), so a
    suggestion engine that only used parser output falls silent on exactly the plans worth
    inspecting - which is what it did on the first run.
    """
    result = parse(text)
    # Where the parser has positively identified the token as something that is *not* an
    # area, take its word for it. The fallback below exists to rescue areas the parser could
    # not read, not to overrule the ones it read correctly: an elevation mark "+46.290"
    # contains a perfectly good-looking decimal and is never a room area.
    if result.kind is DimKind.DOOR_WINDOW_CODE or result.fmt in _NOT_AN_AREA:
        return []
    if (
        result.kind is DimKind.AREA
        and result.area_m2 is not None
        and result.plausible
        # A bare integer with no unit is not an area on these plans - it is a door code, a
        # wall run or a room number. Offering "7" as the kitchen's area just invites a
        # wrong keystroke.
        and not result.area_needs_corroboration
    ):
        return [result.area_m2]

    values: list[float] = []
    spans: list[tuple[int, int]] = []
    for match in _DECIMAL.finditer(text):
        try:
            values.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue
        spans.append(match.span())
    for match in _INT_WITH_UNIT.finditer(text):
        # Skip an integer that is really the tail of a decimal already captured: in
        # "$8,1 m^{2}$" the unit pattern otherwise matches "1 m^{2" and offers 1 m2.
        start, end = match.span(1)
        if any(s <= start < e or s < end <= e for s, e in spans):
            continue
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return [v for v in values if MIN_AREA <= v <= MAX_AREA]


def suggest(
    tokens: list[OcrToken], label: str, used: set[int]
) -> list[tuple[float, int, str]]:
    """Areas printed near a token bearing this label, nearest first.

    Returns (value, token index, raw text). Token indices already consumed by an earlier
    room are skipped, so the second MH gets the second area rather than repeating the first.
    """
    out: list[tuple[float, float, int, str]] = []
    for label_index in label_token_indices(tokens, label):
        anchor = tokens[label_index]
        # The label token may carry its own area ("MH 11.7").
        if label_index not in used:
            for value in readable_numbers(anchor.text):
                out.append((0.0, value, label_index, anchor.text))
        height = max(1.0, min(anchor.bbox[2] - anchor.bbox[0],
                              anchor.bbox[3] - anchor.bbox[1]))
        for i, token in enumerate(tokens):
            if i == label_index or i in used:
                continue
            values = readable_numbers(token.text)
            if not values:
                continue
            distance = weighted_distance(anchor.bbox, token.bbox)
            if distance > height * MAX_LABEL_HEIGHTS:
                continue
            for value in values:
                out.append((distance, value, i, token.text))
    out.sort(key=lambda t: t[0])
    seen: set[tuple[float, int]] = set()
    unique = []
    for _distance, value, index, text in out:
        if (value, index) in seen:
            continue
        seen.add((value, index))
        unique.append((value, index, text))
    return unique[:3]


def all_parseable_areas(tokens: list[OcrToken]) -> list[tuple[int, str, float]]:
    found = []
    for i, token in enumerate(tokens):
        result = parse(token.text)
        if result.kind is DimKind.AREA and result.area_m2 is not None and result.plausible:
            found.append((i, token.text, result.area_m2))
    return found


def load_record(plan_id: str) -> dict:
    path = OUT_DIR_JSON / f"{plan_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_record(plan_id: str, payload: dict) -> None:
    OUT_DIR_JSON.mkdir(parents=True, exist_ok=True)
    payload["printed_area_count"] = sum(
        1 for r in payload["rooms"] if r.get("decided") and r.get("area_m2") is not None
    )
    payload["decided_count"] = sum(1 for r in payload["rooms"] if r.get("decided"))
    payload["complete"] = payload["decided_count"] == len(payload["rooms"])
    payload["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (OUT_DIR_JSON / f"{plan_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        # Non-interactive (no TTY). Treat as quit rather than looping forever on "".
        print("\n(no input available - run with a TTY: docker compose run --rm ...)")
        raise SystemExit(1) from None


def label_plan(plan_id: str, dataset: Path, review: bool = False) -> str:
    directory = plan_dir(dataset, f"/high_quality_architectural/{plan_id}/")
    tokens = load_tokens(plan_id)
    svg = parse_model_svg(directory / "model.svg")
    image_path = render(plan_id, directory / "F1_scaled.png", tokens)
    used: set[int] = set()

    payload = load_record(plan_id)
    if not payload or review:
        existing = {r["svg_index"]: r for r in payload.get("rooms", [])}
        payload = {
            "plan_id": plan_id,
            "folder": "high_quality_architectural",
            "method": "read off the rendered page, one room at a time",
            "note": (
                "Areas are what the DRAWING prints, not model.svg polygon areas. A null "
                "area_m2 with decided=true means the drawing prints no area for that room."
            ),
            "rooms": [],
        }
        for i, room in enumerate(svg.rooms):
            prior = existing.get(i, {}) if review else {}
            payload["rooms"].append({
                "svg_index": i,
                "svg_class": room.svg_class,
                "svg_name": room.name,
                "svg_polygon_area_m2": round(room.area_m2, 2),
                "label": prior.get("label", room.name),
                "area_m2": prior.get("area_m2"),
                "decided": prior.get("decided", False),
                "area_source": prior.get("area_source"),
                "label_source": prior.get("label_source", "svg"),
            })
        save_record(plan_id, payload)

    rooms = payload["rooms"]
    print("\n" + "=" * 78)
    print(f"PLAN {plan_id}   {len(rooms)} rooms in model.svg   {len(tokens)} OCR tokens")
    print(f"open this while labelling:  {image_path}")
    print("=" * 78)
    print(HELP)

    i = 0
    while i < len(rooms):
        room = rooms[i]
        if room["decided"] and not review:
            i += 1
            continue

        options = suggest(tokens, room["svg_name"], used)

        print(f"\n--- room {i + 1}/{len(rooms)} " + "-" * 50)
        print(f"  SVG:      {room['svg_name']!r}  ({room['svg_class']})")
        print(f"  polygon:  {room['svg_polygon_area_m2']} m2   "
              "(annotation, not the printed figure)")
        label_ids = label_token_indices(tokens, room["svg_name"])
        print("  label on page: " + (
            ", ".join(f'id{j} "{tokens[j].text}"' for j in label_ids[:4])
            if label_ids else "(no token matches this label)"
        ))
        if options:
            for n, (value, index, text) in enumerate(options):
                pct = 100 * abs(value - room["svg_polygon_area_m2"]) / max(
                    room["svg_polygon_area_m2"], 1e-6
                )
                marker = "SUGGEST:" if n == 0 else f"      {n + 1}:"
                print(f'  {marker} {value:g} m2  from id{index} "{text}"  '
                      f"(polygon differs by {pct:.0f}%)")
            if len(options) > 1:
                print("            press 2 or 3 to take an alternative")
        else:
            # No anchor to measure from - usually because OCR read the label rotated
            # ("OH" comes back as "HO"). Rather than guess, show what areas are still
            # unclaimed on the page so the value can be found by eye and typed.
            print("  SUGGEST:  (none - no token matches this label; OCR may have read it "
                  "rotated)")
            spare = [
                f'id{j} "{t.text}" -> {v:g}'
                for j, t in enumerate(tokens)
                if j not in used
                for v in readable_numbers(t.text)[:1]
            ]
            if spare:
                print("  unclaimed areas on the page: " + ", ".join(spare[:5]))
        if room["decided"]:
            print(f"  current:  {room['area_m2']} ({room['area_source']})")

        answer = ask("  > ")

        if answer == "q":
            save_record(plan_id, payload)
            return "quit"
        if answer == "?":
            print(HELP)
            continue
        if answer == "s":
            i += 1
            continue
        if answer == "b":
            i = max(0, i - 1)
            continue
        if answer == "l":
            new_label = ask(f"    label as printed [{room['label']}]: ").strip()
            if new_label:
                room["label"] = new_label
                room["label_source"] = "typed"
                save_record(plan_id, payload)
            continue
        if answer == "n":
            room["area_m2"] = None
            room["area_source"] = "read_as_not_printed"
            room["decided"] = True
        elif answer in ("", "y", "2", "3"):
            pick = {"": 0, "y": 0, "2": 1, "3": 2}[answer]
            if pick >= len(options):
                print("    no such suggestion - type a number, or 'n' if none printed")
                continue
            value, index, raw = options[pick]
            room["area_m2"] = value
            room["area_source"] = "confirmed_suggestion"
            room["area_token"] = {"index": index, "raw": raw}
            used.add(index)
            room["decided"] = True
        else:
            try:
                room["area_m2"] = float(answer.replace(",", "."))
            except ValueError:
                print("    not a number. Enter/y, a number, n, l, s, b, ? or q")
                continue
            room["area_source"] = "typed"
            room["decided"] = True

        save_record(plan_id, payload)
        i += 1

    # Anything printed that no room claimed - the SVG does miss rooms.
    claimed = {r["area_m2"] for r in rooms if r["area_m2"] is not None}
    leftovers = [
        (idx, text, value)
        for idx, text, value in all_parseable_areas(tokens)
        if not any(abs(value - c) <= 0.01 * max(c, 1e-6) for c in claimed)
    ]
    extras = payload.setdefault("extra_areas", [])
    if leftovers and not payload.get("extras_reviewed"):
        print(f"\n--- {len(leftovers)} parseable area(s) on the page that no room claimed ---")
        print("  For each: 'y' if the drawing really prints it as a room area, else Enter.")
        for idx, text, value in leftovers:
            reply = ask(f'  id{idx} "{text}" -> {value:g} m2   [y/N/q] ')
            if reply == "q":
                save_record(plan_id, payload)
                return "quit"
            if reply == "y":
                label = ask("    label it is printed under (blank if none): ").strip()
                extras.append({
                    "token_index": idx, "raw": text, "area_m2": value,
                    "label": label or None, "area_source": "typed_extra",
                })
        payload["extras_reviewed"] = True

    save_record(plan_id, payload)
    printed = payload["printed_area_count"] + len(extras)
    print(f"\n  plan {plan_id} done: {printed} printed area(s) recorded")
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="*", help="plan ids to walk through")
    parser.add_argument("--dataset", type=Path, default=Path("/data/cubicasa5k"))
    parser.add_argument("--review", help="re-open a finished plan for correction")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    args = parser.parse_args()

    plan_ids = [args.review] if args.review else args.plans
    if not plan_ids and not args.status:
        parser.error("give one or more plan ids, or --review <id>, or --status")

    if args.status:
        print(f"{'plan':>7} {'decided':>9} {'printed':>8}  state")
        known = sorted(p.stem for p in OUT_DIR_JSON.glob("*.json")) if OUT_DIR_JSON.exists() else []
        if not known:
            print("  (nothing recorded yet)")
            return 0
        for plan_id in known:
            payload = load_record(plan_id)
            if not payload:
                print(f"{plan_id:>7} {'-':>9} {'-':>8}  not started")
                continue
            state = "complete" if payload.get("complete") else "in progress"
            print(f"{plan_id:>7} {payload['decided_count']:>4}/{len(payload['rooms']):<4} "
                  f"{payload['printed_area_count']:>8}  {state}")
        return 0

    for plan_id in plan_ids:
        payload = load_record(plan_id)
        if payload.get("complete") and not args.review:
            print(f"plan {plan_id}: already complete, skipping "
                  f"(use --review {plan_id} to reopen)")
            continue
        if label_plan(plan_id, args.dataset, review=bool(args.review)) == "quit":
            print("\nsaved. Re-run to continue where you stopped.")
            return 0

    done = [p for p in plan_ids if load_record(p).get("complete")]
    print(f"\n{len(done)} of {len(plan_ids)} plan(s) complete. "
          f"Written to {OUT_DIR_JSON}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
