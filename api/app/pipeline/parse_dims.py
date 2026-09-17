"""Dimension and area parsing for Finnish floor plans.

Returns a typed result with a reason, so eval can report accuracy per format and so a
caller can tell "this is not a dimension" apart from "this is a dimension I failed to read".

What is actually printed on these plans
---------------------------------------
Finnish plans label a room with a name and a single **area**, not a width x length pair::

    MH          <- makuuhuone (bedroom)
    11.7        <- 11.7 m squared

so ``AREA`` is the common case and ``area_m2`` is the field that matters. Width/length pairs
do occur and are parsed when present, but they are never invented from an area.

The hard part is everything that *looks* like a dimension and is not:

``9X21``
    A Finnish door/window size code in decimetres (0.9 m x 2.1 m). These blanket
    architectural plans - far outnumbering real room dimensions - and are the single
    largest false-positive source. Classified as ``DOOR_WINDOW_CODE`` and excluded.

``19940``, ``2750``
    Wall run dimensions in millimetres along the page margin. Not room areas.

``+46.290``
    Floor elevation marks.

``1:100``
    Drawing scale.

``4H K KH WC 90 M2``
    An apartment summary (4 rooms + kitchen + bath + WC, 90 m squared) describing the whole
    unit rather than any one room.

Ambiguity that cannot be resolved from the token alone
------------------------------------------------------
``3 x 4`` with no unit and no decimal separator is indistinguishable from a door code by
shape. On Finnish plans a bare integer pair where both sides are <= 30 is overwhelmingly a
door/window code, so that is how it is classified. ``raw`` is always preserved so a later
stage with more context (neighbouring tokens, room polygon area) can override. This choice
is deliberate and is unit-tested both ways.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.pipeline.room_types import (
    UNKNOWN,
    is_known_label,
    normalise_label,
    primary_type,
    resolve_label,
)

# Areas outside this range are flagged implausible. A 1 m2 room is smaller than a shower
# tray; 500 m2 is larger than any single room in a residential plan.
MIN_PLAUSIBLE_AREA_M2 = 1.0
MAX_PLAUSIBLE_AREA_M2 = 500.0

# Area formats carrying no corroborating evidence: no decimal separator, no area unit.
# See ParseResult.area_needs_corroboration.
_WEAK_AREA_FMTS = frozenset({"area_bare_integer", "label_area_integer"})

# Door/window codes are decimetres. A 30 dm (3 m) leaf is already unusually large, so a
# bare integer pair above this is not a door code.
MAX_DOOR_CODE_DM = 30

# Below this, a bare integer pair is not a millimetre dimension (400 mm = 40 cm).
MIN_MM_VALUE = 400


class DimKind(str, Enum):
    AREA = "area"
    DIMENSION_PAIR = "dimension_pair"
    DOOR_WINDOW_CODE = "door_window_code"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParseResult:
    kind: DimKind
    raw: str
    reason: str
    area_m2: float | None = None
    width_m: float | None = None
    length_m: float | None = None
    # Populated when the token carries its own label, e.g. "MH 11.7".
    label: str | None = None
    room_type: str | None = None
    # Areas outside the plausible range are returned, not dropped, but flagged.
    plausible: bool = True
    # Which pattern matched, for per-format accuracy reporting in eval.
    fmt: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_dimension(self) -> bool:
        """True only for things that describe a room's size."""
        return self.kind in (DimKind.AREA, DimKind.DIMENSION_PAIR)

    @property
    def area_needs_corroboration(self) -> bool:
        """True for an area read off a digit string with no decimal point and no unit.

        Every one of the 29 areas on the hand-verified 6-plan set is printed as a decimal
        ("MH 11.7"), and none as a bare integer. A bare integer that lands near a room
        label is therefore far more likely to be a room number, a millimetre wall run, a
        door code or an OCR fragment of the apartment summary than a real area. The value
        is still returned and still tagged, so eval can report it; it is the *pairing*
        stage that decides whether the page carries enough evidence to trust it.
        """
        return self.kind is DimKind.AREA and self.fmt in _WEAK_AREA_FMTS


def _unknown(raw: str, reason: str, fmt: str = "") -> ParseResult:
    return ParseResult(kind=DimKind.UNKNOWN, raw=raw, reason=reason, fmt=fmt)


# --- number normalisation -------------------------------------------------------------

# Finnish uses the comma as decimal separator ("12,5"); the scanned plans inspected so far
# print a period. Both are accepted. A comma is always a decimal separator here, never a
# thousands separator: Finnish groups thousands with a space, and no plausible plan value
# needs grouping anyway.
_NUM = r"\d+(?:[.,]\d+)?"


def _to_float(text: str) -> float:
    return float(text.replace(",", "."))


def _has_decimal(text: str) -> bool:
    return "." in text or "," in text


# --- unit detection -------------------------------------------------------------------

_AREA_UNIT = r"(?:m\s*[²2]|M\s*[²2])"
_LEN_UNIT = r"(?:mm|cm|m)"

_CLEAN_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _CLEAN_RE.sub(" ", text.strip())


# --- rejection patterns ---------------------------------------------------------------

_SCALE_RE = re.compile(r"^1\s*[:/]\s*\d{1,4}$")
_ELEVATION_RE = re.compile(r"^[+-]\s*\d{1,3}[.,]\d{2,3}$")
_SHEET_RE = re.compile(r"^[A-Za-z]{1,3}[-_ ]?\d{1,4}$|^\d{1,3}\s*/\s*\d{1,3}$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# "4H K KH WC 90 M2" - an apartment summary, not a room.
_APARTMENT_SUMMARY_RE = re.compile(
    rf"^\s*\d+\s*H\b.*?({_NUM})\s*{_AREA_UNIT}\s*$", re.IGNORECASE
)

# The Finnish apartment *type* code: "3H+K+S", "2H+KK", "3H+KT+S" - a room count, then
# "+"-joined codes for kitchen/sauna. It labels the whole unit, and the area printed beside
# it is the unit total, never a room's.
#
# Matched separately from _APARTMENT_SUMMARY_RE because OCR routinely splits the code away
# from its area into different tokens, and mangles the characters either side: on plan 2504
# "3H+KT+S" came back as "BH+KT+$" (3 read as B, S as $) with "61,0m2" as its own token.
# The "+"-joined skeleton survives that mangling even when the individual letters do not, so
# the leading and trailing characters are deliberately permissive - only the structure is
# load-bearing.
_APARTMENT_CODE_RE = re.compile(
    r"[0-9A-Za-z]\s*H\s*\+\s*[A-Za-z$§]{1,3}(?:\s*\+\s*[A-Za-z$§]{1,3})*"
)


def looks_like_apartment_code(text: str) -> bool:
    """True when a token carries the Finnish whole-apartment type code ("3H+KT+S").

    Used by the pairing stage to suppress an area printed alongside it: that number is the
    apartment total, not a room area.
    """
    return bool(_APARTMENT_CODE_RE.search(text))


# --- pair patterns --------------------------------------------------------------------

_PAIR_RE = re.compile(
    rf"^({_NUM})\s*(?:{_LEN_UNIT})?\s*[x×X]\s*({_NUM})\s*({_LEN_UNIT})?\s*$"
)

# Imperial, kept because the parser should not be dataset-specific:
#   12'6" x 10'      12' - 6" x 10' - 0"
_FEET_IN = r"(\d+)\s*'\s*(?:-\s*)?(?:(\d+)\s*\")?"
_IMPERIAL_PAIR_RE = re.compile(rf"^{_FEET_IN}\s*[x×X]\s*{_FEET_IN}$")

# Bare area, optionally with a unit: "11.7", "11,7 m²", "90 M2"
_AREA_RE = re.compile(rf"^({_NUM})\s*({_AREA_UNIT})?$")

# Label plus area in a single token: "MH 11.7", "KHH 10,8 m²"
# The label may be several words ("TYO HUONE") because OCR inserts spaces inside a word,
# and may carry an area unit. Everything before the final number is the label.
_LABEL_AREA_RE = re.compile(
    rf"^([A-Za-zÄÖÅäöå][A-Za-zÄÖÅäöå.+/ ]*?)\s+({_NUM})\s*(?:{_AREA_UNIT})?$"
)


def _feet_inches_to_m(feet: str, inches: str | None) -> float:
    total_in = int(feet) * 12 + (int(inches) if inches else 0)
    return total_in * 0.0254


def _area_result(
    value: float, raw: str, fmt: str, *, label: str | None = None, explicit_unit: bool
) -> ParseResult:
    plausible = MIN_PLAUSIBLE_AREA_M2 <= value <= MAX_PLAUSIBLE_AREA_M2
    warnings: list[str] = []
    reason = "bare number read as an area" if not explicit_unit else "area with explicit unit"
    if not plausible:
        side = "below" if value < MIN_PLAUSIBLE_AREA_M2 else "above"
        warnings.append(
            f"area {value:g} m2 is {side} the plausible range "
            f"[{MIN_PLAUSIBLE_AREA_M2:g}, {MAX_PLAUSIBLE_AREA_M2:g}]"
        )
        reason = f"{reason}; implausible magnitude"
    return ParseResult(
        kind=DimKind.AREA,
        raw=raw,
        reason=reason,
        area_m2=round(value, 3),
        label=label,
        room_type=primary_type(label) if label else None,
        plausible=plausible,
        fmt=fmt,
        warnings=tuple(warnings),
    )


def _classify_pair(a_text: str, b_text: str, unit: str | None, raw: str) -> ParseResult:
    """Decide what a ``A x B`` token means."""
    a, b = _to_float(a_text), _to_float(b_text)
    decimals = _has_decimal(a_text) or _has_decimal(b_text)
    unit = (unit or "").lower()

    if a <= 0 or b <= 0:
        return _unknown(raw, "pair contains a zero or negative side", fmt="pair")

    # Explicit unit settles it outright.
    if unit == "mm":
        return _pair_result(a / 1000, b / 1000, raw, "mm_pair_explicit", "millimetre pair")
    if unit == "cm":
        return _pair_result(a / 100, b / 100, raw, "cm_pair", "centimetre pair")
    if unit == "m":
        return _pair_result(a, b, raw, "m_pair_explicit", "metre pair with explicit unit")

    # A decimal separator means metres: door codes are always whole decimetres.
    if decimals:
        return _pair_result(a, b, raw, "m_pair_decimal", "decimal pair read as metres")

    # Bare integers from here on.
    if a <= MAX_DOOR_CODE_DM and b <= MAX_DOOR_CODE_DM:
        return ParseResult(
            kind=DimKind.DOOR_WINDOW_CODE,
            raw=raw,
            reason=(
                f"integer pair with both sides <= {MAX_DOOR_CODE_DM} and no unit: "
                "Finnish door/window size code in decimetres, not a room dimension"
            ),
            width_m=round(a / 10, 3),
            length_m=round(b / 10, 3),
            fmt="door_window_code",
        )

    if a >= MIN_MM_VALUE and b >= MIN_MM_VALUE:
        return _pair_result(a / 1000, b / 1000, raw, "mm_pair", "large integer pair read as mm")

    return _unknown(
        raw,
        f"integer pair {a:g}x{b:g} matches neither a door code "
        f"(both <= {MAX_DOOR_CODE_DM}) nor a millimetre pair (both >= {MIN_MM_VALUE})",
        fmt="pair",
    )


def _pair_result(width: float, length: float, raw: str, fmt: str, reason: str) -> ParseResult:
    area = width * length
    plausible = MIN_PLAUSIBLE_AREA_M2 <= area <= MAX_PLAUSIBLE_AREA_M2
    warnings: list[str] = []
    if not plausible:
        warnings.append(f"implied area {area:.3g} m2 is outside the plausible range")
    return ParseResult(
        kind=DimKind.DIMENSION_PAIR,
        raw=raw,
        reason=reason,
        width_m=round(width, 3),
        length_m=round(length, 3),
        area_m2=round(area, 3),
        plausible=plausible,
        fmt=fmt,
        warnings=tuple(warnings),
    )


def parse(text: str) -> ParseResult:
    """Classify a single OCR token.

    Never raises: malformed input returns ``UNKNOWN`` with a reason.
    """
    if text is None:
        return _unknown("", "empty token")
    raw = _clean(text)
    if not raw:
        return _unknown(raw, "empty token")

    # --- explicit rejections, checked before anything numeric ---
    if _SCALE_RE.match(raw):
        return _unknown(raw, "drawing scale label", fmt="scale")
    if _ELEVATION_RE.match(raw):
        return _unknown(raw, "floor elevation mark", fmt="elevation")
    if _APARTMENT_SUMMARY_RE.match(raw):
        return _unknown(
            raw, "apartment summary describing the whole unit, not a room", fmt="apartment_summary"
        )
    if _SHEET_RE.match(raw) and not is_known_label(raw):
        return _unknown(raw, "sheet or drawing number", fmt="sheet_number")

    # --- pairs ---
    imperial = _IMPERIAL_PAIR_RE.match(raw)
    if imperial:
        w = _feet_inches_to_m(imperial.group(1), imperial.group(2))
        length = _feet_inches_to_m(imperial.group(3), imperial.group(4))
        if w <= 0 or length <= 0:
            return _unknown(raw, "imperial pair contains a zero side", fmt="imperial_pair")
        return _pair_result(w, length, raw, "imperial_pair", "imperial feet/inches pair")

    pair = _PAIR_RE.match(raw)
    if pair:
        return _classify_pair(pair.group(1), pair.group(2), pair.group(3), raw)

    # --- label + area in one token ---
    label_area = _LABEL_AREA_RE.match(raw)
    if label_area:
        label, number = label_area.group(1), label_area.group(2)
        resolved = resolve_label(label)
        if resolved is not None:
            explicit_unit = bool(re.search(_AREA_UNIT, raw))
            # "MH 11.7" is an area; "MH 1" is almost always a room *number* - this dataset
            # numbers repeated rooms MH1/MH2, AH 1, AH 3. Tag the two apart so the pairing
            # stage can demand corroboration for the integer form without losing the tag
            # that per-format eval reporting needs.
            fmt = (
                "label_area"
                if explicit_unit or _has_decimal(number)
                else "label_area_integer"
            )
            return _area_result(
                _to_float(number), raw, fmt, label=label, explicit_unit=explicit_unit,
            )
        return _unknown(
            raw,
            f"looks like a label+area pair but {normalise_label(label)!r} "
            "is not a known Finnish room label",
            fmt="label_area",
        )

    # --- bare area ---
    area = _AREA_RE.match(raw)
    if area:
        number, unit = area.group(1), area.group(2)
        if unit:
            return _area_result(_to_float(number), raw, "area_unit", explicit_unit=True)
        # A bare integer with no decimal and no unit is usually not an area: it is a
        # millimetre wall run, a year or a sheet number. Areas on these plans are printed
        # with a decimal ("11.7"), so require one unless the value is small.
        if _YEAR_RE.match(raw):
            return _unknown(
                raw,
                "bare 4-digit integer in year range: a year, sheet number or millimetre "
                "wall run, but not a room area",
                fmt="bare_integer",
            )
        if not _has_decimal(number):
            value = _to_float(number)
            if value > MAX_PLAUSIBLE_AREA_M2:
                return _unknown(
                    raw,
                    f"bare integer {value:g} exceeds the plausible area range; "
                    "most likely a millimetre wall run",
                    fmt="bare_integer",
                )
            return _area_result(value, raw, "area_bare_integer", explicit_unit=False)
        return _area_result(_to_float(number), raw, "area_bare", explicit_unit=False)

    # --- a plain room label on its own ---
    if is_known_label(raw):
        return ParseResult(
            kind=DimKind.UNKNOWN,
            raw=raw,
            reason="room label with no dimension attached",
            label=raw,
            room_type=primary_type(raw),
            fmt="label_only",
        )

    return _unknown(raw, "no dimension, area or code pattern matched")


def parse_label_and_area(label_text: str, area_text: str) -> ParseResult:
    """Combine a label and an area found in two separate tokens.

    Association by bounding-box proximity happens in the pipeline, not here - this function
    only interprets a pairing that has already been decided.
    """
    result = parse(area_text)
    if result.kind is not DimKind.AREA:
        return result
    label = _clean(label_text)
    room_type = primary_type(label)
    return ParseResult(
        kind=result.kind,
        raw=f"{label} {result.raw}".strip(),
        reason=f"{result.reason}; label associated from a neighbouring token",
        area_m2=result.area_m2,
        width_m=result.width_m,
        length_m=result.length_m,
        label=label or None,
        room_type=room_type if room_type != UNKNOWN else None,
        plausible=result.plausible,
        fmt=f"{result.fmt}+assoc",
        warnings=result.warnings,
    )


def pattern_score(text: str) -> float:
    """How strongly ``text`` looks like a meaningful plan token, in [0, 1].

    Used by the OCR orientation merge to choose between competing readings of one box: a
    mirrored reading such as ``IZX6`` scores 0 where ``9X21`` scores high. Kept here so the
    vocabulary lives in one place rather than being restated in ocr.py.
    """
    result = parse(text)
    if result.kind is DimKind.DOOR_WINDOW_CODE:
        return 0.9
    if result.kind in (DimKind.AREA, DimKind.DIMENSION_PAIR):
        return 0.9 if result.plausible else 0.4
    if result.fmt in ("elevation", "scale", "apartment_summary"):
        return 0.8
    if result.fmt == "bare_integer":
        # A millimetre wall run is a real, correctly-read token even though it is not a
        # room dimension - "19940" should still beat a mirrored "0466L".
        return 0.6
    return 0.0
