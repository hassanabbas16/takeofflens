"""OCR wrapper: returns {text, confidence, bbox} in original page coordinates.

Why this module is more than a thin call to PaddleOCR
-----------------------------------------------------
Floor plan text is not document text. Two measured problems drove the design, both from a
sweep on high_quality_architectural/1191 (a plan whose printed labels are known):

1. **Labels are set at 90 degrees.** Architectural plans rotate room labels to fit inside
   narrow rooms, so a single upright pass misses most of them. PaddleOCR's textline
   orientation classifier corrects a line it has already *detected*, but the detector
   itself finds far fewer vertical lines to begin with. Worse, on vertical text it often
   picks the wrong 180-degree flip and returns confidently mirrored garbage - "9X21" comes
   back as "IZX6", "KHH" as "HHM", "5515" as "GISS".

   Fix: OCR the page at 0/90/180/270, map every box back to original page coordinates, and
   merge. Measured on plan 1191: one pass recovered 2/12 labels, 0+90 recovered 7/12, and
   all four recovered 8/12.

2. **A bigger detection input makes recall worse.** Raising text_det_limit_side_len from
   1536 to 2400 dropped the token count from 47 to 26. PaddleOCR's detector resizes to its
   limit regardless, and these pages are mostly empty white space with small text, so
   feeding it more pixels makes the glyphs smaller relative to the receptive field rather
   than larger. The default 1536 is kept deliberately - it is not an oversight.

Merging duplicates: the same text is usually found in several orientation passes, one of
which read it correctly and the rest mirrored. We keep the highest-confidence reading among
overlapping boxes, because a correctly-oriented read tends to score higher than a mirrored
one ("9X21" at 0.789 beat "IZX6" at 0.721). This is a heuristic, and it is the most likely
source of residual garbage tokens; eval reports the failure cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2

_ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    bbox: BBox
    angle: int  # orientation pass this token came from, kept for debugging

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "angle": self.angle,
        }


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    if angle not in _ROTATE_FLAGS:
        raise ValueError(f"unsupported rotation {angle}; expected 0/90/180/270")
    return cv2.rotate(img, _ROTATE_FLAGS[angle])


def unrotate_point(x: float, y: float, angle: int, height: int, width: int) -> tuple[float, float]:
    """Map a point from a rotated image back to original coordinates.

    ``height``/``width`` are the ORIGINAL page dimensions. Derived from how cv2.rotate
    remaps indices; round-tripped in tests/test_ocr_geometry.py because an off-by-one here
    silently shifts every overlay in the viewer.
    """
    if angle == 0:
        return x, y
    if angle == 90:
        return y, (height - 1) - x
    if angle == 180:
        return (width - 1) - x, (height - 1) - y
    if angle == 270:
        return (width - 1) - y, x
    raise ValueError(f"unsupported rotation {angle}")


def unrotate_bbox(poly: np.ndarray, angle: int, height: int, width: int) -> BBox:
    """Map a detection polygon back to an axis-aligned bbox in original coordinates."""
    pts = [unrotate_point(float(p[0]), float(p[1]), angle, height, width) for p in poly]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (
        max(0, int(round(min(xs)))),
        max(0, int(round(min(ys)))),
        min(width, int(round(max(xs)))),
        min(height, int(round(max(ys)))),
    )


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate(tokens: list[OcrToken], iou_threshold: float) -> list[OcrToken]:
    """Keep the highest-confidence reading among overlapping boxes.

    Greedy and confidence-descending: the standard NMS shape, but selecting a *reading*
    rather than suppressing a duplicate detection.
    """
    kept: list[OcrToken] = []
    for token in sorted(tokens, key=lambda t: t.confidence, reverse=True):
        if any(iou(token.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(token)
    # Reading order: top-to-bottom, then left-to-right.
    return sorted(kept, key=lambda t: (t.bbox[1], t.bbox[0]))


@lru_cache(maxsize=4)
def _get_engine(
    lang: str, det_limit_side_len: int, det_limit_type: str
) -> Any:  # noqa: ANN401 - PaddleOCR ships no type stubs to annotate against
    # Imported lazily so importing this module (and the whole app) does not pull in
    # Paddle. Keeps API startup and non-OCR tests fast.
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        # Page-level orientation and unwarping are off: rotation is handled by the
        # multi-orientation merge below, and these add latency without helping on plans.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_det_limit_side_len=det_limit_side_len,
        text_det_limit_type=det_limit_type,
    )


def run_ocr(
    image: np.ndarray,
    *,
    lang: str = "en",
    orientations: tuple[int, ...] | list[int] = (0, 90, 180, 270),
    det_limit_side_len: int = 1536,
    det_limit_type: str = "max",
    min_confidence: float = 0.5,
    dedup_iou: float = 0.5,
) -> list[OcrToken]:
    """OCR ``image`` at each orientation and merge into original page coordinates."""
    if image.ndim != 3:
        # PaddleOCR's detector unpacks three values from img.shape and crashes on a
        # single-channel array, with an error that points nowhere near the cause.
        raise ValueError("run_ocr expects a 3-channel BGR image")

    height, width = image.shape[:2]
    engine = _get_engine(lang, det_limit_side_len, det_limit_type)

    tokens: list[OcrToken] = []
    for angle in orientations:
        rotated = rotate_image(image, angle)
        try:
            results = engine.predict(rotated)
        except Exception:
            # One bad orientation must not lose the other three.
            logger.exception("OCR failed at %d degrees", angle)
            continue
        for result in results:
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polys = result.get("rec_polys", result.get("dt_polys", []))
            for text, score, poly in zip(texts, scores, polys, strict=False):
                cleaned = text.strip()
                if not cleaned or float(score) < min_confidence:
                    continue
                tokens.append(
                    OcrToken(
                        text=cleaned,
                        confidence=float(score),
                        bbox=unrotate_bbox(np.asarray(poly), angle, height, width),
                        angle=angle,
                    )
                )

    return deduplicate(tokens, dedup_iou)


def draw_tokens(image: np.ndarray, tokens: list[OcrToken]) -> np.ndarray:
    """Draw token boxes and readings for the debug image."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for token in tokens:
        x1, y1, x2, y2 = token.bbox
        # Colour by confidence - green high, amber mid, red low - so it is obvious at a
        # glance which readings to distrust.
        if token.confidence >= 0.9:
            colour = (0, 170, 0)
        elif token.confidence >= 0.7:
            colour = (0, 165, 255)
        else:
            colour = (0, 0, 220)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            canvas,
            f"{token.text} {token.confidence:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
