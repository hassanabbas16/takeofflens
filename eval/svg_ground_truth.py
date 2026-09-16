"""Extract room ground truth from a CubiCasa ``model.svg``.

This is the core of Tier 2: automatic ground truth, free, for the whole dataset.

Important framing: ``model.svg`` is CubiCasa's *annotation* of the plan, not the plan
itself. Its dimension labels are ``display: none`` and never appear in the page image, so
these values are never treated as OCR ground truth for text. What they give is the room
inventory: type, Finnish label, polygon and area.

Scale is a constant 100 SVG units per metre (median 100.02 over 5127 room edges across 200
test plans), so a room's area is the shoelace area of its polygon divided by 10000. Each
plan's own median ratio is recomputed here and reported so a deviating plan is flagged
rather than silently trusted.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from html import unescape
from pathlib import Path

# 100 SVG units = 1 metre.
UNITS_PER_METRE = 100.0

_SPACE_RE = re.compile(r'<g[^>]*class="Space ([^"]+)"[^>]*>(.*?)(?=<g[^>]*class="Space |\Z)', re.S)
_POLY_RE = re.compile(r'<polygon points="([^"]+)"')
_NAME_RE = re.compile(r'class="TextLabel NameLabel"[^>]*>\s*<text[^>]*>([^<]*)</text>')
_DIM_RE = re.compile(r'class="TextLabel DimensionMeasureLabel"[^>]*>(.*?)</g>', re.S)
_TEXT_RE = re.compile(r">([^<>]+)</text>")
_METRIC_RE = re.compile(r"([\d.]+)\s*m\s*x\s*([\d.]+)\s*m")


@dataclass(frozen=True)
class SvgRoom:
    svg_class: str          # e.g. "Bedroom", "Outdoor Balcony"
    name: str               # Finnish label, entity-unescaped, e.g. "MH", "KEITTIÖ"
    area_m2: float          # shoelace area / 10000
    bbox: tuple[float, float, float, float]
    labelled_w_m: float | None  # from the (hidden) dimension label, when present
    labelled_h_m: float | None

    @property
    def room_type(self) -> str:
        """Coarse type: the first word of the SVG class, lower-cased."""
        return self.svg_class.split()[0].lower() if self.svg_class else "undefined"


@dataclass(frozen=True)
class SvgPlan:
    path: Path
    rooms: list[SvgRoom]
    scale_ratio: float | None   # measured units-per-metre for this plan
    scale_ok: bool

    @property
    def room_count(self) -> int:
        return len(self.rooms)


def _shoelace(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _parse_points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for chunk in raw.strip().split():
        if "," not in chunk:
            continue
        x_str, _, y_str = chunk.partition(",")
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def parse_model_svg(path: Path) -> SvgPlan:
    """Parse a ``model.svg`` into rooms with areas in square metres."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    rooms: list[SvgRoom] = []
    ratios: list[float] = []

    for match in _SPACE_RE.finditer(text):
        svg_class = match.group(1).strip()
        body = match.group(2)

        poly = _POLY_RE.search(body)
        if not poly:
            continue
        points = _parse_points(poly.group(1))
        if len(points) < 3:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        name_match = _NAME_RE.search(body)
        name = unescape(name_match.group(1)).strip() if name_match else ""

        labelled_w = labelled_h = None
        dim = _DIM_RE.search(body)
        if dim:
            for chunk in _TEXT_RE.findall(dim.group(1)):
                metric = _METRIC_RE.match(chunk.strip())
                if metric:
                    labelled_w = float(metric.group(1))
                    labelled_h = float(metric.group(2))
                    break

        # Scale evidence: bbox extent against the labelled dimension. L-shaped rooms make
        # this disagree, hence the median rather than the mean.
        if labelled_w and labelled_w > 0.2:
            ratios.append((bbox[2] - bbox[0]) / labelled_w)
        if labelled_h and labelled_h > 0.2:
            ratios.append((bbox[3] - bbox[1]) / labelled_h)

        rooms.append(
            SvgRoom(
                svg_class=svg_class,
                name=name,
                area_m2=round(_shoelace(points) / (UNITS_PER_METRE**2), 3),
                bbox=bbox,
                labelled_w_m=labelled_w,
                labelled_h_m=labelled_h,
            )
        )

    ratio = statistics.median(ratios) if ratios else None
    scale_ok = ratio is not None and abs(ratio - UNITS_PER_METRE) <= 1.0
    return SvgPlan(path=path, rooms=rooms, scale_ratio=ratio, scale_ok=scale_ok)


def plan_dir(dataset_dir: Path, entry: str) -> Path:
    """Resolve a split-file entry such as ``/high_quality_architectural/1191/``."""
    return Path(dataset_dir) / entry.strip().strip("/")


def read_split(dataset_dir: Path, split: str) -> list[str]:
    lines = (Path(dataset_dir) / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]
