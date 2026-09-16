"""Pydantic schemas.

These are the single source of truth for the extraction contract: the same models generate
the structured-output schema sent to Claude, validate the response, and shape the API payload.

The model is constrained to this schema via ``client.messages.parse(output_format=...)``.
Optional values are expressed as nullable types (``float | None``) rather than omitted
fields, so the model must explicitly say "no area here" instead of leaving the key out -
which is also what grounding needs in order to tell a missing area from an unstated one.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.room_types import ROOM_TYPES


def _room_type_members() -> dict[str, str]:
    """Enum members from the Finnish->English map, plus an explicit escape hatch.

    Derived from the lexicon rather than hand-listed so the two cannot drift. ``other`` is
    added so the model has somewhere to put a genuine room type we do not cover - forcing a
    wrong choice from a closed list would be worse than knowing it is uncovered.
    """
    members = {value.upper(): value for value in sorted(set(ROOM_TYPES.values()))}
    members["OTHER"] = "other"
    return members


RoomType = Enum("RoomType", _room_type_members(), type=str)
RoomType.__doc__ = "English room type, mapped from the Finnish label."


class Source(str, Enum):
    OCR_LLM = "ocr+llm"
    VLM = "vlm"
    HYBRID = "hybrid"
    DETECTOR = "detector"  # Phase 8


class ExtractedRoom(BaseModel):
    """One room as extracted by a model. Shared by all three approaches."""

    model_config = ConfigDict(extra="forbid")

    label_raw: str = Field(
        description=(
            "The room label exactly as it appears on the drawing, in Finnish, "
            "with original spelling and case. Do not translate it."
        )
    )
    room_type: RoomType = Field(
        description="The English room type for this label. Use 'other' if none fits."
    )
    area_m2: float | None = Field(
        description=(
            "Floor area in square metres, as printed on the drawing. "
            "Null if no area is printed for this room. Never estimate or calculate it."
        )
    )
    width_m: float | None = Field(
        description=(
            "Width in metres, only if the drawing prints an explicit width x length pair "
            "for this room. Null otherwise. Never derive this from an area."
        )
    )
    length_m: float | None = Field(
        description=(
            "Length in metres, only if the drawing prints an explicit width x length pair. "
            "Null otherwise. Never derive this from an area."
        )
    )
    source_token_ids: list[int] | None = Field(
        description=(
            "IDs of the OCR tokens this room was read from - the label token and the area "
            "token. Required when OCR tokens were provided. Null only when working from "
            "the image alone."
        )
    )
    confidence: float = Field(
        description="How confident you are in this room, from 0.0 to 1.0."
    )


class PlanExtraction(BaseModel):
    """Top-level extraction result. This is the schema the model is constrained to."""

    model_config = ConfigDict(extra="forbid")

    rooms: list[ExtractedRoom]
    notes: str | None = Field(
        description=(
            "Anything that affected the reading: unreadable text, the page appearing "
            "rotated, or no room labels being present. Null if nothing to report."
        )
    )


# --- grounding -------------------------------------------------------------------------


class GroundingIssue(BaseModel):
    """One reason a room failed grounding. Counted as a hallucination in metrics."""

    room_index: int
    label_raw: str
    kind: str  # unknown_token_id | ungrounded_area | missing_token_ids | ungrounded_label
    detail: str


class GroundedExtraction(BaseModel):
    """An extraction after grounding, with the rejected rooms kept for reporting."""

    rooms: list[ExtractedRoom]
    dropped: list[ExtractedRoom]
    issues: list[GroundingIssue]
    notes: str | None = None

    @property
    def hallucination_count(self) -> int:
        return len(self.issues)


# --- API payloads ----------------------------------------------------------------------


class TokenOut(BaseModel):
    id: int
    text: str
    confidence: float
    bbox: list[int]


class RoomOut(BaseModel):
    id: UUID
    name: str | None
    room_type: str
    width_m: float | None
    length_m: float | None
    area_m2: float | None
    source: str
    raw_text: str | None
    bbox: list[int] | None
    confidence: float | None
    grounded: bool


class PageOut(BaseModel):
    id: UUID
    page_number: int
    width: int
    height: int
    tokens: list[TokenOut] = []
    rooms: list[RoomOut] = []


class PlanOut(BaseModel):
    id: UUID
    filename: str
    status: str
    error: str | None
    created_at: datetime
    pages: list[PageOut] = []


class LlmCallOut(BaseModel):
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
