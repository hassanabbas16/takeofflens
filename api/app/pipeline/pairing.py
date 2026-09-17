"""Pair room labels with nearby area tokens by bounding-box proximity.

Done before the LLM call, not by it. On these plans the label and its area sit one directly
above the other (``MH`` with ``11.7`` beneath it), which is a geometric fact the model
should not have to rediscover from a flat token list - and if it does rediscover it, we have
no way to check it. Passing explicit candidate pairs makes the model's job smaller and its
citations checkable.

Distance is measured centre-to-centre but weighted: vertical separation counts for less than
horizontal, because the area is nearly always directly under or over the label rather than
beside it. Both tokens of a pair must be plausible - a room label and a parseable area -
so door codes and wall runs are never offered as a room's area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.pipeline.ocr import BBox, OcrToken
from app.pipeline.parse_dims import DimKind, ParseResult, looks_like_apartment_code, parse
from app.pipeline.room_types import is_known_label

# Vertical distance is scaled by this before comparing, so a token directly below a label
# is preferred over one the same absolute distance away horizontally.
VERTICAL_WEIGHT = 0.6

# Candidate pairs further apart than this (in multiples of the label's own height) are not
# offered at all. Keeps a label from being paired with an area on the far side of the page.
MAX_DISTANCE_IN_LABEL_HEIGHTS = 6.0


@dataclass(frozen=True)
class TokenRef:
    """An OCR token with the stable per-page id the model will cite."""

    id: int
    text: str
    confidence: float
    bbox: BBox

    @classmethod
    def from_tokens(cls, tokens: list[OcrToken]) -> list[TokenRef]:
        return [
            cls(id=i, text=t.text, confidence=t.confidence, bbox=t.bbox)
            for i, t in enumerate(tokens)
        ]


@dataclass(frozen=True)
class CandidatePair:
    label: TokenRef
    area: TokenRef
    area_m2: float
    distance: float
    # True when one token carried both the label and the area ("khh 6.8").
    combined: bool = False
    # The label text alone, with any area stripped off.
    label_text: str = ""

    def as_prompt_line(self) -> str:
        if self.combined:
            return (
                f'  label id={self.label.id} "{self.label_text}"'
                f'  area from the same token "{self.label.text}" -> {self.area_m2:g} m2'
            )
        return (
            f'  label id={self.label.id} "{self.label_text or self.label.text}"'
            f'  area id={self.area.id} "{self.area.text}" -> {self.area_m2:g} m2'
        )


def _centre(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def weighted_distance(a: BBox, b: BBox) -> float:
    ax, ay = _centre(a)
    bx, by = _centre(b)
    return math.hypot(ax - bx, (ay - by) * VERTICAL_WEIGHT)


def _overlaps(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _label_height(bbox: BBox) -> float:
    # Rotated labels have a tall box, so the "height" of the text is its shorter side.
    return max(1.0, float(min(bbox[2] - bbox[0], bbox[3] - bbox[1])))


def find_candidate_pairs(refs: list[TokenRef]) -> list[CandidatePair]:
    """Pair every plausible room label with its nearest plausible area token.

    One pair per label, and **one label per area**: a printed area belongs to exactly one
    room. Letting several labels share the nearest area produced four rooms with an
    identical 16.0 m2 on a plan that prints no areas at all - a confident fabrication from
    one stray token. When labels contend for an area the closest wins and the others are
    reported with no area, which is the truthful outcome.

    A token that already carries both ("khh 6.8") pairs with **itself**. Before this was
    handled, such a token appeared in both the label list and the area list, was forbidden
    from matching itself, and so was paired with some other room's area - producing a
    confidently wrong pair rather than a missing one. Ten of the twenty-six tokens on one
    real plan were of this form.

    An area also has to be *corroborated* to be offered at all: see the comment on
    ``has_corroborated_area`` below. A page that prints no trustworthy area anywhere
    offers no bare integers as areas, which is what stops stray digits on an area-less
    plan from being reported as room areas.
    """
    parsed = {ref.id: parse(ref.text) for ref in refs}

    # Does this page print any area we can actually trust - one with a decimal separator
    # ("11.7") or an explicit unit ("55 m2")? If it does, a bare integer among them is
    # plausibly just another area. If it does not, every bare integer on the page is
    # something else: a room number, a millimetre wall run, or an OCR fragment of the
    # apartment summary. Plan 2536 prints no areas at all, yet OCR shattered its summary
    # string "4H K KH WC 90 M2" into fragments and two of them ("016", "2") landed near a
    # label and were reported as WC 16.0 and ET 2.0. Requiring a corroborating sibling
    # costs nothing on the hand-verified 6-plan set, where all 29 printed areas are
    # decimals, and removes the fabrications on the four plans that print none.
    has_corroborated_area = any(
        r.kind is DimKind.AREA
        and r.area_m2 is not None
        and r.plausible
        and not r.area_needs_corroboration
        for r in parsed.values()
    )

    # An area printed alongside the whole-apartment type code ("3H+KT+S 61,0 m2") is the
    # unit total, not a room's. OCR splits the code from its number into separate tokens, so
    # the association has to be made geometrically: an area token overlapping the code token
    # belongs to the summary. On plan 2504 this is what handed the living room the
    # apartment's own 61 m2.
    summary_boxes = [r.bbox for r in refs if looks_like_apartment_code(r.text)]

    def in_summary_block(ref: TokenRef) -> bool:
        return any(_overlaps(ref.bbox, box) for box in summary_boxes)

    def area_is_trusted(result: ParseResult, ref: TokenRef) -> bool:
        return (
            result.kind is DimKind.AREA
            and result.area_m2 is not None
            and result.plausible
            and (has_corroborated_area or not result.area_needs_corroboration)
            and not in_summary_block(ref)
        )

    combined: list[CandidatePair] = []
    combined_ids: set[int] = set()
    for ref in refs:
        result = parsed[ref.id]
        if area_is_trusted(result, ref) and result.label:
            combined.append(
                CandidatePair(
                    label=ref, area=ref, area_m2=result.area_m2, distance=0.0,
                    combined=True, label_text=result.label,
                )
            )
            combined_ids.add(ref.id)

    labels = [
        r for r in refs
        if r.id not in combined_ids and (is_known_label(r.text) or parsed[r.id].label)
    ]
    areas: list[tuple[TokenRef, float]] = []
    for ref in refs:
        if ref.id in combined_ids:
            # Already spoken for: this token's area belongs to its own label.
            continue
        result = parsed[ref.id]
        if area_is_trusted(result, ref):
            areas.append((ref, result.area_m2))

    pairs: list[CandidatePair] = list(combined)
    for label in labels:
        limit = _label_height(label.bbox) * MAX_DISTANCE_IN_LABEL_HEIGHTS
        best: tuple[TokenRef, float, float] | None = None
        for area_ref, area_value in areas:
            if area_ref.id == label.id:
                continue
            distance = weighted_distance(label.bbox, area_ref.bbox)
            if distance > limit:
                continue
            if best is None or distance < best[2]:
                best = (area_ref, area_value, distance)
        if best is not None:
            pairs.append(
                CandidatePair(
                    label=label, area=best[0], area_m2=best[1],
                    distance=round(best[2], 1), label_text=label.text,
                )
            )

    return sorted(_resolve_contention(pairs), key=lambda p: p.label.id)


def _resolve_contention(pairs: list[CandidatePair]) -> list[CandidatePair]:
    """Keep at most one pair per area token: the closest label wins.

    A combined token owns its own area outright and never contends.
    """
    winner: dict[int, CandidatePair] = {}
    kept: list[CandidatePair] = []
    for pair in pairs:
        if pair.combined:
            kept.append(pair)
            continue
        current = winner.get(pair.area.id)
        if current is None or pair.distance < current.distance:
            winner[pair.area.id] = pair
    return kept + list(winner.values())


def format_token_list(refs: list[TokenRef]) -> str:
    """The token table sent to the model. Compact: this is most of the prompt cost."""
    lines = []
    for ref in refs:
        x1, y1, x2, y2 = ref.bbox
        lines.append(
            f'  id={ref.id} "{ref.text}" bbox=[{x1},{y1},{x2},{y2}] '
            f"conf={ref.confidence:.2f}"
        )
    return "\n".join(lines)


def format_candidate_pairs(pairs: list[CandidatePair]) -> str:
    if not pairs:
        return "  (none found by proximity)"
    return "\n".join(p.as_prompt_line() for p in pairs)
