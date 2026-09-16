"""Ingest: PDF or image in, normalised page images out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


class IngestError(ValueError):
    """Raised for unreadable or unsupported input, so jobs can fail with a message."""


@dataclass(frozen=True)
class Page:
    page_number: int
    image: np.ndarray  # BGR, uint8

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def _render_pdf(path: Path, dpi: int) -> list[Page]:
    pages: list[Page] = []
    with fitz.open(path) as doc:
        if doc.page_count == 0:
            raise IngestError(f"PDF has no pages: {path.name}")
        for index, page in enumerate(doc):
            # zoom = dpi/72 because PDF user space is 72 units per inch.
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            buf = np.frombuffer(pix.samples, dtype=np.uint8)
            img = buf.reshape(pix.height, pix.width, pix.n)
            # PyMuPDF gives RGB; OpenCV and PaddleOCR both expect BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pages.append(Page(page_number=index + 1, image=img))
    return pages


def _read_image(path: Path) -> list[Page]:
    # imdecode rather than imread: imread silently fails on non-ASCII paths on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IngestError(f"could not decode image: {path.name}")
    return [Page(page_number=1, image=img)]


def ingest(path: Path, dpi: int = 300) -> list[Page]:
    """Load ``path`` into a list of BGR page images.

    PDFs render at ``dpi``; images pass through at native resolution.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(
            f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".pdf":
        return _render_pdf(path, dpi)
    return _read_image(path)
