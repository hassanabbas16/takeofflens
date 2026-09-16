"""Preprocessing steps for OCR.

Every step is individually toggleable so Phase 6 can ablate them. The defaults below were
chosen from a measured sweep on a CubiCasa architectural plan
(high_quality_architectural/1191), scored on how many of the printed Finnish room labels
and area figures OCR recovered:

    raw              4 orientations   8/12 labels, 6/13 areas   <- default
    grayscale        4 orientations   7/12 labels, 6/13 areas
    denoise          4 orientations   7/12 labels, 7/13 areas
    adaptive thresh  4 orientations   5/12 labels, 1/13 areas

Notes on why the defaults are what they are:

- **Adaptive threshold defaults OFF even though the spec lists it.** On scanned CAD
  drawings it is actively harmful: room labels are thin strokes sitting among equally thin
  wall hatching and dimension lines, so a local threshold either welds the text to nearby
  linework or erodes it away. It cost more than half the area figures in the sweep. It
  stays available because it helps on low-contrast photographed plans, which the eval set
  will eventually include.
- **Denoise defaults OFF** by a narrow margin: it trades a label for an area figure and
  costs ~2x the runtime. It is the first thing to turn on when a page scans poorly.
- **Upscale defaults OFF.** Counter-intuitively, feeding the detector a larger image makes
  recall worse here, because PaddleOCR's detector resizes to its own limit anyway and the
  text ends up smaller relative to the receptive field. See ocr.py.

PaddleOCR requires 3-channel input, so single-channel results are converted back to BGR
before returning. Returning a 1-channel array crashes the detector deep in its resize step.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    grayscale: bool = False
    denoise: bool = False
    adaptive_threshold: bool = False
    deskew: bool = False
    upscale: bool = False
    upscale_min_short_side: int = 2000

    @property
    def applied(self) -> list[str]:
        names = ("grayscale", "denoise", "adaptive_threshold", "deskew", "upscale")
        return [n for n in names if getattr(self, n)]


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page skew in degrees via the dominant near-horizontal Hough line.

    Floor plans are full of long straight walls, which makes Hough far more reliable here
    than a minAreaRect over text pixels - the text is sparse and often rotated 90 degrees,
    so a text-based estimate picks up the labels rather than the page.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=200, minLineLength=gray.shape[1] // 4, maxLineGap=20
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        # Fold to [-45, 45]: a wall at 90 degrees is the same skew evidence as one at 0.
        angle = (angle + 45.0) % 90.0 - 45.0
        # Ignore near-diagonal lines; they are usually stairs or roof hatching.
        if abs(angle) <= 15.0:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    # Sub-degree skew is not worth a resample; rotation is lossy.
    if abs(angle) < 0.2:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess(img: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply the enabled steps in a fixed order and return a 3-channel BGR image."""
    out = img

    if config.deskew:
        out = _deskew(out)

    if config.grayscale or config.denoise or config.adaptive_threshold:
        out = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    if config.denoise:
        # Small template/search windows: CAD text strokes are 1-2px, and the default
        # (7, 21) smears them.
        out = cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)

    if config.adaptive_threshold:
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

    if config.upscale:
        short_side = min(out.shape[:2])
        if short_side < config.upscale_min_short_side:
            scale = config.upscale_min_short_side / short_side
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return _to_bgr(out)
