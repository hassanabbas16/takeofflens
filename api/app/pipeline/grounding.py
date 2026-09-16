"""Grounding: check that what the model returned is actually supported by the page.

For ``ocr+llm`` and ``hybrid`` this is a **hard** check, because OCR tokens were supplied and
every claim is therefore checkable:

- every cited ``source_token_ids`` entry must exist on the page
- every non-null ``area_m2`` must match an area that the parser actually found in a token
- a room with no citations at all is ungrounded

For ``vlm`` it is a **soft** signal only. The vision model reads the image directly, so it
can legitimately see an area that OCR missed - and on these plans OCR misses about half of
them. Failing a VLM room for disagreeing with OCR would measure OCR, not the model. Soft
issues are recorded and reported, never dropped.

Anything that fails a hard check is counted as a hallucination in metrics. The default is to
keep the room with ``grounded=False`` rather than delete it, so the failure stays visible and
countable; ``drop=True`` removes them from the returned set as well.
"""

from __future__ import annotations

from app.pipeline.pairing import TokenRef
from app.pipeline.parse_dims import DimKind, parse
from app.schemas import ExtractedRoom, GroundedExtraction, GroundingIssue, PlanExtraction

# An area counts as grounded if it is within this relative tolerance of a parsed token
# value. Not zero, because the model may reasonably normalise "11,7" to 11.7.
AREA_MATCH_TOLERANCE = 0.02


def _parsed_areas(refs: list[TokenRef]) -> list[float]:
    values: list[float] = []
    for ref in refs:
        result = parse(ref.text)
        if result.kind is DimKind.AREA and result.area_m2 is not None:
            values.append(result.area_m2)
        # A dimension pair also justifies an area claim, since the pair implies one.
        elif result.kind is DimKind.DIMENSION_PAIR and result.area_m2 is not None:
            values.append(result.area_m2)
    return values


def _area_is_supported(area: float, available: list[float]) -> bool:
    return any(
        abs(area - value) <= AREA_MATCH_TOLERANCE * max(abs(value), 1e-6) for value in available
    )


def check_grounding(
    extraction: PlanExtraction,
    refs: list[TokenRef],
    *,
    hard: bool = True,
    drop: bool = False,
) -> GroundedExtraction:
    """Validate an extraction against the OCR tokens it claims to come from.

    ``hard=False`` records issues without marking rooms ungrounded - used for ``vlm``.
    """
    valid_ids = {ref.id for ref in refs}
    available_areas = _parsed_areas(refs)

    kept: list[ExtractedRoom] = []
    dropped: list[ExtractedRoom] = []
    issues: list[GroundingIssue] = []

    for index, room in enumerate(extraction.rooms):
        room_issues: list[GroundingIssue] = []

        if hard:
            if not room.source_token_ids:
                room_issues.append(
                    GroundingIssue(
                        room_index=index,
                        label_raw=room.label_raw,
                        kind="missing_token_ids",
                        detail="no source_token_ids cited although OCR tokens were provided",
                    )
                )
            else:
                unknown = [i for i in room.source_token_ids if i not in valid_ids]
                if unknown:
                    room_issues.append(
                        GroundingIssue(
                            room_index=index,
                            label_raw=room.label_raw,
                            kind="unknown_token_id",
                            detail=f"cited token ids that do not exist on this page: {unknown}",
                        )
                    )

        if room.area_m2 is not None and not _area_is_supported(room.area_m2, available_areas):
            room_issues.append(
                GroundingIssue(
                    room_index=index,
                    label_raw=room.label_raw,
                    kind="ungrounded_area",
                    detail=(
                        f"area {room.area_m2:g} m2 does not match any area parsed from an "
                        "OCR token on this page"
                    ),
                )
            )

        issues.extend(room_issues)
        if hard and room_issues:
            dropped.append(room)
            if not drop:
                kept.append(room)
        else:
            kept.append(room)

    return GroundedExtraction(
        rooms=kept, dropped=dropped, issues=issues, notes=extraction.notes
    )


def grounded_room_ids(room: ExtractedRoom) -> str | None:
    if not room.source_token_ids:
        return None
    return ",".join(str(i) for i in room.source_token_ids)


def room_bbox_from_tokens(room: ExtractedRoom, refs: list[TokenRef]) -> list[int] | None:
    """Union of the cited tokens' boxes, so the viewer can highlight the source."""
    if not room.source_token_ids:
        return None
    by_id = {ref.id: ref.bbox for ref in refs}
    boxes = [by_id[i] for i in room.source_token_ids if i in by_id]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
