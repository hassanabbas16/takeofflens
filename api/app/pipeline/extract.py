"""The three extraction approaches, sharing one output schema.

``rules``    OCR + parser + bbox-proximity pairing, no model call at all ($0)
``ocr+llm``  OCR tokens (ids + boxes) + pre-computed candidate pairs -> text model
``vlm``      the page image alone -> vision model
``hybrid``   the page image plus the OCR token list as hints -> vision model

``rules`` is the baseline the paid approaches have to beat. It appears in every results
table, because an LLM approach that does not clearly beat free deterministic machinery is
not earning its cost.

They differ only in what evidence they are given. The schema, the grounding step and the
persistence path are identical, which is the point: the comparison in eval is then about the
evidence, not about three different pipelines.

The prompts state the Finnish conventions explicitly, because a model that does not know
them will read ``9X21`` as a room dimension and ``11,7`` as 117.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.config import get_settings
from app.llm import LlmClient, LlmUsage
from app.pipeline.grounding import check_grounding
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import (
    TokenRef,
    find_candidate_pairs,
    format_candidate_pairs,
    format_token_list,
)
from app.pipeline.room_types import is_known_label, primary_type
from app.schemas import ExtractedRoom, GroundedExtraction, PlanExtraction, RoomType, Source

logger = logging.getLogger(__name__)

# Shared domain briefing. Kept in one constant so all three approaches are told the same
# things about the domain and only differ in the evidence they receive. Sent as the `system`
# prompt rather than inline in the user turn: it is identical across every page, so it sits
# at the front of the prefix where prompt caching can reuse it.
_FINNISH_RULES = """\
These are Finnish residential floor plans. The conventions matter:

- Room labels are Finnish, usually abbreviations: MH (makuuhuone, bedroom), OH (olohuone,
  living room), K or KEITTIO (kitchen), ET (eteinen, entry), KHH (utility), KPH/KH
  (bathroom), PH/PSH (washroom), WC, VH (walk-in closet), TK (draught lobby), VAR (storage),
  PARVEKE (balcony), TERASSI (terrace), ULKOTILA (outdoor). Labels may also be spelled out
  in full and may be lowercase.
- The number printed under or beside a room label is an AREA in square metres, not a
  width x length pair. "MH 11.7" means a bedroom of 11.7 m2.
- Decimal separators may be a comma or a period: "11,7" and "11.7" are the same number.
- Codes like 9X21, 10x21, 14X12 are door and window sizes in decimetres. They are NOT room
  dimensions and must never be reported as a room's area, width or length.
- Large bare integers along the page margin (19940, 2750, 3970) are wall runs in
  millimetres, not room areas.
- Marks like +46.290 are floor elevations.
- A string like "4H K KH WC 90 M2" is a summary of the whole apartment, not a room.

Rules for your answer:
- Report only rooms you can actually see evidence for. Do not invent rooms.
- area_m2 must be a number printed on the drawing. If no area is printed for a room, use
  null. Never estimate, calculate or infer an area.
- width_m and length_m are only for an explicit printed "width x length" pair. Otherwise
  null. Never derive them from an area.
- label_raw must be the label exactly as printed, in Finnish, not translated."""

_CITATION_RULES = """\
- For every room, set source_token_ids to the ids of the OCR tokens you used: the label
  token, and the area token if there is one. Every id you cite must come from the list.
  Do not invent ids. A room you cannot tie to a token should not be reported."""


@dataclass
class ExtractionOutcome:
    """What one approach produced for one page."""

    source: Source
    grounded: GroundedExtraction | None
    usage: LlmUsage
    refs: list[TokenRef] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.grounded is not None

    @property
    def rooms(self) -> list[Any]:
        return self.grounded.rooms if self.grounded else []

    @property
    def hallucinations(self) -> int:
        return self.grounded.hallucination_count if self.grounded else 0


def encode_page_image(image: np.ndarray, max_px: int, jpeg_quality: int) -> tuple[str, int, int]:
    """Downscale and JPEG-encode a page for a vision call.

    Cost scales with image tokens, which scale with pixels, so the long edge is capped.
    JPEG rather than PNG: these are mostly-white line drawings and PNG of a 1536px scan is
    several times larger for no benefit the model can use.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_px:
        scale = max_px / longest
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError("failed to JPEG-encode the page image")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return encoded, image.shape[1], image.shape[0]


def system_prompt() -> str:
    """The shared domain briefing, so the batch path sends exactly what the sync path does."""
    return _FINNISH_RULES


def image_content(encoded: str) -> dict[str, Any]:
    # Anthropic takes the base64 payload and media type as structured fields, not as a
    # data: URL.
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded},
    }


def text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _run(
    client: LlmClient, model: str, messages: list[dict[str, Any]], purpose: str
) -> tuple[PlanExtraction | None, LlmUsage]:
    settings = get_settings()
    result = client.parse(
        model=model,
        messages=messages,
        schema=PlanExtraction,
        system=_FINNISH_RULES,
        temperature=settings.anthropic_temperature,
        purpose=purpose,
    )
    return result.parsed, result.usage


# --- ocr+llm ---------------------------------------------------------------------------


def build_ocr_llm_prompt(refs: list[TokenRef]) -> str:
    """The user turn for the ocr+llm approach.

    Split out from the call so the synchronous path and the Batch API path send byte-for-byte
    the same prompt - otherwise the batch run would not be measuring the same thing.
    """
    pairs = find_candidate_pairs(refs)
    return f"""{_CITATION_RULES}

You are given the OCR output of one floor plan page. The page was OCR'd at four rotations
and merged, so some readings may still be garbled; ignore tokens that are not meaningful.

OCR tokens (id, text, bounding box as [x1,y1,x2,y2], confidence):
{format_token_list(refs)}

Candidate label/area pairs, computed geometrically by proximity. These are suggestions from
box positions alone, not ground truth - accept the ones that make sense and ignore the rest:
{format_candidate_pairs(pairs)}

Return every room you can identify."""


def extract_ocr_llm(
    tokens: list[OcrToken], client: LlmClient, model: str | None = None
) -> ExtractionOutcome:
    settings = get_settings()
    model = model or settings.anthropic_text_model
    refs = TokenRef.from_tokens(tokens)
    prompt = build_ocr_llm_prompt(refs)

    parsed, usage = _run(
        client, model, [{"role": "user", "content": prompt}], purpose="classify"
    )
    if parsed is None:
        return ExtractionOutcome(Source.OCR_LLM, None, usage, refs, usage.error)
    return ExtractionOutcome(Source.OCR_LLM, ground_ocr_llm(parsed, refs), usage, refs)


def ground_ocr_llm(parsed: PlanExtraction, refs: list[TokenRef]) -> GroundedExtraction:
    return check_grounding(parsed, refs, hard=True, drop=False)


# --- vlm -------------------------------------------------------------------------------


def extract_vlm(
    image: np.ndarray,
    client: LlmClient,
    model: str | None = None,
    tokens: list[OcrToken] | None = None,
) -> ExtractionOutcome:
    """Vision model on the page image alone.

    ``tokens`` are not shown to the model; they are used only to soft-check the areas it
    returns, so the report can say how often it agreed with OCR.
    """
    settings = get_settings()
    model = model or settings.anthropic_vision_model
    encoded, width, height = encode_page_image(
        image, settings.vlm_max_image_px, settings.vlm_jpeg_quality
    )

    prompt = build_vlm_prompt(width, height)

    parsed, usage = _run(
        client,
        model,
        [{"role": "user", "content": [text_content(prompt), image_content(encoded)]}],
        purpose="vlm",
    )
    if parsed is None:
        return ExtractionOutcome(Source.VLM, None, usage, [], usage.error)

    refs = TokenRef.from_tokens(tokens) if tokens else []
    return ExtractionOutcome(Source.VLM, ground_vlm(parsed, refs), usage, refs)


def build_vlm_prompt(width: int, height: int) -> str:
    """The user turn for the vlm approach."""
    return f"""You are looking at one floor plan page, {width}x{height} pixels.

Important: room labels on these plans are frequently ROTATED 90 degrees, so text often runs
vertically up or down the page rather than left to right. Read it in whatever orientation it
is set. Some pages are scanned, hand-lettered or annotated in pen.

Some plans print no room areas at all - only labels. That is normal. In that case return the
rooms with area_m2 as null rather than guessing a number.

Set source_token_ids to null: no OCR tokens were provided.

Return every room you can identify."""


def ground_vlm(parsed: PlanExtraction, refs: list[TokenRef]) -> GroundedExtraction:
    """Soft check only: the vision model can legitimately read an area OCR missed, so a
    disagreement with OCR measures OCR rather than the model."""
    return check_grounding(parsed, refs, hard=False, drop=False)


# --- hybrid ----------------------------------------------------------------------------


def extract_hybrid(
    image: np.ndarray, tokens: list[OcrToken], client: LlmClient, model: str | None = None
) -> ExtractionOutcome:
    settings = get_settings()
    model = model or settings.anthropic_vision_model
    refs = TokenRef.from_tokens(tokens)
    encoded, width, height = encode_page_image(
        image, settings.vlm_max_image_px, settings.vlm_jpeg_quality
    )
    prompt = build_hybrid_prompt(refs, width, height)

    parsed, usage = _run(
        client,
        model,
        [{"role": "user", "content": [text_content(prompt), image_content(encoded)]}],
        purpose="hybrid",
    )
    if parsed is None:
        return ExtractionOutcome(Source.HYBRID, None, usage, refs, usage.error)
    return ExtractionOutcome(Source.HYBRID, ground_hybrid(parsed, refs), usage, refs)


def build_hybrid_prompt(refs: list[TokenRef], width: int, height: int) -> str:
    """The user turn for the hybrid approach."""
    pairs = find_candidate_pairs(refs)
    return f"""You are looking at one floor plan page, {width}x{height} pixels, together
with the OCR output for that same page.

Room labels on these plans are frequently ROTATED 90 degrees. The OCR ran at four rotations
and merged its results, so some readings are garbled or mirrored. The image is authoritative:
where OCR disagrees with what you can see, trust your own reading of the image.

OCR tokens (id, text, bounding box as [x1,y1,x2,y2], confidence):
{format_token_list(refs)}

Candidate label/area pairs computed geometrically by proximity:
{format_candidate_pairs(pairs)}

{_CITATION_RULES}
- If you read a room from the image that OCR did not capture at all, report it with
  source_token_ids set to null rather than citing an unrelated token.

Return every room you can identify."""


def ground_hybrid(parsed: PlanExtraction, refs: list[TokenRef]) -> GroundedExtraction:
    """Hard check on ids, because tokens were supplied and citations are checkable.

    A room the model explicitly marks as image-only (null ids) is allowed by the prompt, so
    missing ids are not a hallucination here - only wrong ids and unsupported areas are.
    """
    grounded = check_grounding(parsed, refs, hard=True, drop=False)
    return GroundedExtraction(
        rooms=grounded.rooms,
        dropped=grounded.dropped,
        issues=[i for i in grounded.issues if i.kind != "missing_token_ids"],
        notes=grounded.notes,
    )


# --- rules (no LLM) ----------------------------------------------------------------------


def extract_rules(tokens: list[OcrToken]) -> ExtractionOutcome:
    """OCR + parser + bbox-proximity pairing, with no model call at all.

    This is the floor every paid approach has to beat. It costs nothing, runs in under a
    millisecond, and is fully deterministic - so if an LLM approach is not clearly ahead of
    it, the LLM is not earning its cost. Including it in every results table keeps that
    comparison honest rather than implicit.

    It is pure machinery already built for other reasons: the Finnish lexicon decides what
    is a room label, parse_dims decides what is an area (and excludes door codes), and the
    pairing module associates the two by weighted bounding-box proximity. Anything needing
    judgement - a garbled reading, a label with no nearby area, an ambiguous pairing - it
    simply does not report.
    """
    refs = TokenRef.from_tokens(tokens)
    pairs = find_candidate_pairs(refs)
    paired_label_ids = {pair.label.id for pair in pairs}

    rooms: list[ExtractedRoom] = []

    for pair in pairs:
        # label_text is the label alone; pair.label.text may be "khh 6.8" with the area
        # glued on, which would make the label unrecognisable downstream.
        label_text = pair.label_text or pair.label.text
        room_type = primary_type(label_text)
        rooms.append(
            ExtractedRoom(
                label_raw=label_text,
                room_type=_as_room_type(room_type),
                area_m2=pair.area_m2,
                # Never derived: a pair is only reported when the page prints one.
                width_m=None,
                length_m=None,
                source_token_ids=sorted({pair.label.id, pair.area.id}),
                # Confidence is the OCR confidence of the weaker of the two tokens; there is
                # no model opinion to report and inventing one would be dishonest.
                confidence=round(min(pair.label.confidence, pair.area.confidence), 4),
            )
        )

    # Labels with no nearby area are still rooms - they just have no area printed.
    for ref in refs:
        if ref.id in paired_label_ids or not is_known_label(ref.text):
            continue
        rooms.append(
            ExtractedRoom(
                label_raw=ref.text,
                room_type=_as_room_type(primary_type(ref.text)),
                area_m2=None,
                width_m=None,
                length_m=None,
                source_token_ids=[ref.id],
                confidence=round(ref.confidence, 4),
            )
        )

    extraction = PlanExtraction(
        rooms=rooms,
        notes=f"rules approach: {len(pairs)} label/area pairs from {len(refs)} OCR tokens",
    )
    # Grounded by construction - every value came from a token - but run the same check so
    # the column means the same thing in every row of the results table.
    grounded = check_grounding(extraction, refs, hard=True, drop=False)
    usage = LlmUsage(model="none (rules)", ok=True)
    return ExtractionOutcome(Source.RULES, grounded, usage, refs)


def _as_room_type(value: str) -> RoomType:
    try:
        return RoomType(value)
    except ValueError:
        return RoomType("other")
