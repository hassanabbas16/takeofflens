"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[uuid.UUID] = _uuid_pk()
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # pending | processing | done | failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    pages: Mapped[list[Page]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="Page.page_number"
    )


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    plan_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # Storage key, not a filesystem path, so an S3 backend can replace local disk.
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    plan: Mapped[Plan] = relationship(back_populates="pages")
    ocr_tokens: Mapped[list[OcrTokenRow]] = relationship(
        back_populates="page", cascade="all, delete-orphan", order_by="OcrTokenRow.token_index"
    )
    rooms: Mapped[list[Room]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    llm_calls: Mapped[list[LlmCall]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )


class OcrTokenRow(Base):
    __tablename__ = "ocr_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    page_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False
    )
    # Stable per-page index. This is the id the LLM cites in source_token_ids, so it must
    # be small, dense and reproducible - a UUID would waste prompt tokens and invite
    # transcription errors.
    token_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    angle: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    page: Mapped[Page] = relationship(back_populates="ocr_tokens")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


Index("ix_ocr_tokens_page_index", OcrTokenRow.page_id, OcrTokenRow.token_index, unique=True)


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[uuid.UUID] = _uuid_pk()
    page_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    room_type: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    # area_m2 is the primary extracted field: Finnish plans print an area, not a pair.
    # width/length stay null unless a pair is genuinely printed, and are never derived.
    width_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    length_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    area_m2: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ocr+llm | vlm | hybrid | detector
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    x1: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y1: Mapped[int | None] = mapped_column(Integer, nullable=True)
    x2: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y2: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Comma-separated token indices the model cited, kept for auditing a result later.
    source_token_ids: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # False when the room failed the grounding check but was kept for reporting.
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    page: Mapped[Page] = relationship(back_populates="rooms")

    @property
    def bbox(self) -> list[int] | None:
        if self.x1 is None or self.y1 is None or self.x2 is None or self.y2 is None:
            return None
        return [self.x1, self.y1, self.x2, self.y2]


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[uuid.UUID] = _uuid_pk()
    page_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=True
    )
    # classify | vlm | hybrid, plus ":retry" when this was the second attempt
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Anthropic reports cache reads and writes separately from input_tokens, and prices
    # them differently, so both are stored rather than folded into the input count.
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    page: Mapped[Page] = relationship(back_populates="llm_calls")
