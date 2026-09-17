"""Unit tests for the dimension/area parser.

Organised by the thing being proven, because the door/window-code exclusion in particular
is a claim the README makes and needs to be provably true.
"""

import pytest

from app.pipeline.parse_dims import (
    MAX_PLAUSIBLE_AREA_M2,
    MIN_PLAUSIBLE_AREA_M2,
    DimKind,
    parse,
    parse_label_and_area,
    pattern_score,
)

# --- bare areas -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("11.7", 11.7),
        ("11,7", 11.7),
        ("11.7 m²", 11.7),
        ("11,7m2", 11.7),
        ("11,7 M2", 11.7),
        ("90 M2", 90.0),
        ("90M²", 90.0),
        ("37.5", 37.5),
        ("4.0", 4.0),
        ("1.5", 1.5),
        ("22,6 m 2", 22.6),
    ],
)
def test_bare_areas(text, expected):
    result = parse(text)
    assert result.kind is DimKind.AREA, result.reason
    assert result.area_m2 == pytest.approx(expected)
    assert result.plausible
    # Areas must never invent a width or length.
    assert result.width_m is None
    assert result.length_m is None


def test_comma_and_period_agree():
    assert parse("12,5").area_m2 == parse("12.5").area_m2


def test_small_bare_integer_is_an_area():
    result = parse("90")
    assert result.kind is DimKind.AREA
    assert result.area_m2 == pytest.approx(90.0)


# --- label + area ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "label", "area", "room_type"),
    [
        ("MH 11.7", "MH", 11.7, "bedroom"),
        ("MH 11,7", "MH", 11.7, "bedroom"),
        ("KHH 10,8 m²", "KHH", 10.8, "utility"),
        ("OH 22.6", "OH", 22.6, "living_room"),
        ("WC 1.5", "WC", 1.5, "wc"),
        ("PSH 4.0", "PSH", 4.0, "washroom"),
    ],
)
def test_label_area_single_token(text, label, area, room_type):
    result = parse(text)
    assert result.kind is DimKind.AREA
    assert result.label == label
    assert result.area_m2 == pytest.approx(area)
    assert result.room_type == room_type


def test_label_area_rejects_unknown_label():
    result = parse("ZZQ 11.7")
    assert result.kind is DimKind.UNKNOWN
    assert "not a known Finnish room label" in result.reason


def test_parse_label_and_area_from_separate_tokens():
    result = parse_label_and_area("MH", "11,7")
    assert result.kind is DimKind.AREA
    assert result.label == "MH"
    assert result.room_type == "bedroom"
    assert result.area_m2 == pytest.approx(11.7)
    assert "associated from a neighbouring token" in result.reason


def test_parse_label_and_area_propagates_non_area():
    result = parse_label_and_area("MH", "9X21")
    assert result.kind is DimKind.DOOR_WINDOW_CODE


def test_label_only_token_carries_room_type_but_no_area():
    result = parse("KHH")
    assert result.kind is DimKind.UNKNOWN
    assert result.room_type == "utility"
    assert result.area_m2 is None
    assert "no dimension" in result.reason


# --- width x length pairs -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "width", "length"),
    [
        ("3.5x4.2", 3.5, 4.2),
        ("3,5 x 4,2", 3.5, 4.2),
        ("3.5 X 4.2", 3.5, 4.2),
        ("3.5m x 4.2m", 3.5, 4.2),
        ("3500 x 4200", 3.5, 4.2),
        ("3500mm x 4200mm", 3.5, 4.2),
        ("3.5 × 4.2", 3.5, 4.2),
        ("6.86 m x 5.70 m", 6.86, 5.70),
    ],
)
def test_dimension_pairs(text, width, length):
    result = parse(text)
    assert result.kind is DimKind.DIMENSION_PAIR, result.reason
    assert result.width_m == pytest.approx(width, abs=1e-3)
    assert result.length_m == pytest.approx(length, abs=1e-3)
    assert result.area_m2 == pytest.approx(width * length, abs=1e-2)


@pytest.mark.parametrize(
    ("text", "width", "length"),
    [
        ("12'6\" x 10'", 3.8100, 3.0480),
        ("12' - 6\" x 10' - 0\"", 3.8100, 3.0480),
    ],
)
def test_imperial_pairs(text, width, length):
    result = parse(text)
    assert result.kind is DimKind.DIMENSION_PAIR, result.reason
    assert result.width_m == pytest.approx(width, abs=1e-3)
    assert result.length_m == pytest.approx(length, abs=1e-3)


# --- door and window codes ------------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["9X21", "9x21", "10x21", "10X21", "12X12", "14X12", "8X21", "14X16", "12X16", "6X12"]
)
def test_door_window_codes_are_classified_not_read_as_dimensions(code):
    """The README claims these are excluded from room dimensions. Prove it."""
    result = parse(code)
    assert result.kind is DimKind.DOOR_WINDOW_CODE, result.reason
    assert result.is_dimension is False
    assert result.area_m2 is None


def test_door_code_still_exposes_its_physical_size():
    # 9x21 dm = 0.9 m x 2.1 m, which a door schedule would want later.
    result = parse("9X21")
    assert result.width_m == pytest.approx(0.9)
    assert result.length_m == pytest.approx(2.1)


def test_door_code_reason_explains_the_exclusion():
    assert "decimetres" in parse("9X21").reason


def test_decimal_pair_is_not_a_door_code():
    """The decimal separator is what distinguishes 9.0x21.0 metres from the code 9X21."""
    result = parse("9.0x21.0")
    assert result.kind is DimKind.DIMENSION_PAIR


def test_large_integer_pair_is_millimetres_not_a_door_code():
    result = parse("3500 x 4200")
    assert result.kind is DimKind.DIMENSION_PAIR
    assert result.width_m == pytest.approx(3.5)


def test_ambiguous_small_integer_pair_resolves_to_door_code():
    """Documented, deliberate choice: bare "3 x 4" is far more often a door code here."""
    result = parse("3 x 4")
    assert result.kind is DimKind.DOOR_WINDOW_CODE
    assert result.raw == "3 x 4"  # raw preserved so a later stage can override


def test_integer_pair_in_no_mans_land_is_unknown():
    result = parse("50 x 60")
    assert result.kind is DimKind.UNKNOWN
    assert "neither a door code" in result.reason


# --- rejections -----------------------------------------------------------------------


@pytest.mark.parametrize("text", ["1:100", "1:50", "1 : 200", "1/100"])
def test_scale_labels_rejected(text):
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN
    assert result.fmt == "scale"


@pytest.mark.parametrize("text", ["+46.290", "+46.950", "-2,500"])
def test_elevation_marks_rejected(text):
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN
    assert result.fmt == "elevation"


@pytest.mark.parametrize("text", ["1990", "2024", "1974"])
def test_years_rejected(text):
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN
    assert result.fmt == "bare_integer"
    assert "year" in result.reason


@pytest.mark.parametrize("text", ["A-101", "A101", "3/5", "DWG 12"])
def test_sheet_numbers_rejected(text):
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN, result.reason


@pytest.mark.parametrize("text", ["19940", "11000", "8940", "3970", "2750", "5515"])
def test_millimetre_wall_runs_are_not_areas(text):
    """Margin wall dimensions are real numbers but not room areas."""
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN
    assert result.area_m2 is None


def test_apartment_summary_rejected():
    result = parse("4H K KH WC 90 M2")
    assert result.kind is DimKind.UNKNOWN
    assert result.fmt == "apartment_summary"
    assert "whole unit" in result.reason


# --- plausibility ---------------------------------------------------------------------


def test_area_below_range_is_flagged_not_dropped():
    result = parse("0,4 m²")
    assert result.kind is DimKind.AREA
    assert result.plausible is False
    assert result.area_m2 == pytest.approx(0.4)
    assert any("below the plausible range" in w for w in result.warnings)


def test_area_above_range_is_flagged_not_dropped():
    result = parse("640 m²")
    assert result.kind is DimKind.AREA
    assert result.plausible is False
    assert any("above the plausible range" in w for w in result.warnings)


def test_plausible_boundaries_inclusive():
    assert parse(f"{MIN_PLAUSIBLE_AREA_M2:g} m²").plausible
    assert parse(f"{MAX_PLAUSIBLE_AREA_M2:g} m²").plausible


def test_dimension_pair_implausible_area_flagged():
    result = parse("40.0 x 40.0")
    assert result.kind is DimKind.DIMENSION_PAIR
    assert result.plausible is False


# --- malformed input ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["", "   ", "x", "xx", "IZX6", "GISS", "TrOHUONE", "¤", "--", "0k9", "?!", "m²"]
)
def test_malformed_input_returns_unknown_without_raising(text):
    result = parse(text)
    assert result.kind is DimKind.UNKNOWN
    assert result.reason


def test_parse_none_is_safe():
    assert parse(None).kind is DimKind.UNKNOWN


def test_zero_sided_pair_rejected():
    assert parse("0x21").kind is DimKind.UNKNOWN
    assert parse("0.0 x 4.2").kind is DimKind.UNKNOWN


def test_whitespace_is_normalised():
    assert parse("  11,7   m²  ").area_m2 == pytest.approx(11.7)


# --- pattern_score, used by the OCR orientation merge ---------------------------------


def test_pattern_score_prefers_real_readings_over_mirrored_garbage():
    assert pattern_score("9X21") > pattern_score("IZX6")
    assert pattern_score("11.7") > pattern_score("L'P!")
    assert pattern_score("19940") > pattern_score("0466L")


def test_pattern_score_zero_for_nonsense():
    assert pattern_score("GISS") == 0.0
    assert pattern_score("") == 0.0


# --- OCR area-unit repair ----------------------------------------------------------------
#
# Every string here is a real read from the 50-plan Tier 3 run. The recogniser renders the
# superscript in "m2" as LaTeX-like markup on 29 of those 50 plans.


@pytest.mark.parametrize(
    ("text", "area"),
    [
        ("3,3 m^{2}$", 3.3),
        ("6,7 m^{2}$", 6.7),
        ("15.5 m^2}$", 15.5),
        ("$12,3 m^{}$", 12.3),    # the 2 is lost entirely, only the braces survive
        ("11,5m^{2", 11.5),       # truncated, no closing brace
        ("2,3 m{2", 2.3),         # caret lost
        ("36.0M^{2}$", 36.0),     # upper-case M
        ("9,5m^2}$", 9.5),
        ("6.89$", 6.89),          # only the math delimiter survived
    ],
)
def test_mangled_area_unit_is_repaired_and_parsed(text, area):
    result = parse(text)
    assert result.kind is DimKind.AREA
    assert result.area_m2 == pytest.approx(area)


def test_a_repaired_token_reports_the_original_text_and_says_so():
    """The viewer and failure cases must show what OCR produced, not a cleaned-up version."""
    result = parse("3,3 m^{2}$")
    assert result.raw == "3,3 m^{2}$"
    assert any("area unit repaired" in w for w in result.warnings)


def test_an_unrepaired_token_carries_no_repair_warning():
    result = parse("MH 11.7")
    assert result.raw == "MH 11.7"
    assert not any("repaired" in w for w in result.warnings)


@pytest.mark.parametrize(
    "text",
    [
        "17.6m",      # a bare trailing m could be a length; inferring an area is a guess
        "9X21",       # door code
        "+46.290",    # elevation
        "1:100",      # scale
        "19940",      # millimetre wall run
    ],
)
def test_repair_does_not_touch_things_that_are_not_areas(text):
    assert parse(text).kind is not DimKind.AREA


def test_repair_is_idempotent():
    from app.pipeline.parse_dims import normalise_area_unit

    once = normalise_area_unit("3,3 m^{2}$")
    assert normalise_area_unit(once) == once == "3,3 m2"
