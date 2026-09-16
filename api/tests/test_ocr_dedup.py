"""Tests for the orientation-merge dedup.

Every contested pair here was observed in a real run on
high_quality_architectural/1191, with the real confidences and boxes. The point is that
confidence alone picks the wrong reading in each case.
"""

import pytest

from app.pipeline.ocr import (
    OcrToken,
    box_shape,
    deduplicate,
    orientation_is_plausible,
    reading_key,
    vocabulary_score,
)


def tok(text, conf, bbox, angle):
    return OcrToken(text=text, confidence=conf, bbox=bbox, angle=angle)


def pick(tokens, rule="tall_only"):
    kept = deduplicate(tokens, iou_threshold=0.5, rule=rule)
    assert len(kept) == 1, [t.text for t in kept]
    return kept[0].text


# --- the observed contested pairs -----------------------------------------------------


def test_oh_beats_ho():
    """OH/HO: a real Finnish label against its mirror, where the mirror scored higher."""
    assert pick([
        tok("HO", 0.694, (915, 1014, 938, 1049), 270),
        tok("OH", 0.600, (916, 1013, 937, 1050), 90),
    ]) == "OH"


def test_tyohuone_beats_mirrored_variant():
    assert pick([
        tok("TrOHUONE", 0.940, (265, 951, 292, 1050), 180),
        tok("TYOHUONE", 0.880, (266, 950, 291, 1049), 90),
    ]) == "TYOHUONE"


def test_9x21_beats_izx6():
    """Confidence alone would be right here, but the vocabulary must agree."""
    assert pick([
        tok("9X21", 0.823, (418, 590, 441, 640), 90),
        tok("1ZX6", 0.713, (418, 590, 441, 640), 270),
    ]) == "9X21"


def test_9x21_beats_higher_confidence_lexs():
    """The case confidence gets wrong: LEXS at 0.943 against the correct 9X21 at 0.895."""
    assert pick([
        tok("LEXS", 0.943, (537, 524, 592, 551), 90),
        tok("9X21", 0.895, (537, 524, 592, 551), 180),
    ]) == "9X21"


def test_5515_beats_giss():
    assert pick([
        tok("GISS", 0.709, (357, 1503, 414, 1536), 90),
        tok("5515", 0.650, (358, 1504, 413, 1535), 270),
    ]) == "5515"


def test_1049_beats_mirrored_efol():
    """Tall box: the 90 reading is correct and the 180/270 readings are mirrored."""
    assert pick([
        tok("1049", 0.996, (2272, 268, 2301, 324), 90),
        tok("EFOL", 0.865, (2272, 268, 2301, 324), 180),
        tok("6F01", 0.744, (2272, 268, 2301, 324), 270),
    ]) == "1049"


def test_6x12_beats_0x12_with_zero_side():
    """0X12 has a zero side, so it is not a valid door code and must lose."""
    assert pick([
        tok("0X12", 0.927, (84, 271, 111, 325), 180),
        tok("6X12", 0.828, (84, 271, 111, 325), 0),
    ]) == "6X12"


def test_mh_survives_despite_coming_from_a_filtered_orientation():
    """The case that rules out a hard orientation filter.

    MH is read correctly from the 180 pass on a TALL box. Orientation plausibility says
    180 is implausible there, so if orientation outranked vocabulary the correct reading
    would be discarded in favour of "HN".
    """
    assert pick([
        tok("MH", 0.946, (920, 508, 938, 538), 180),
        tok("HN", 0.632, (920, 508, 938, 538), 270),
        tok("Hn", 0.553, (920, 508, 938, 538), 90),
    ]) == "MH"


def test_khh_beats_hhh():
    assert pick([
        tok("KHH", 0.994, (1793, 491, 1812, 534), 180),
        tok("HHH", 0.758, (1793, 491, 1812, 534), 270),
    ]) == "KHH"


def test_area_reading_beats_mirrored_punctuation():
    assert pick([
        tok("14.1", 0.954, (936, 502, 954, 540), 90),
        tok("L'P!", 0.552, (936, 502, 954, 540), 270),
    ]) == "14.1"


# --- orientation as a tie-break, not a filter -----------------------------------------


def test_orientation_breaks_ties_between_equally_wordless_readings():
    """Two readings with no vocabulary support: orientation should decide."""
    tall = (100, 100, 120, 160)
    assert pick([
        tok("qqqq", 0.90, tall, 0),
        tok("wwww", 0.85, tall, 90),
    ]) == "wwww"


def test_confidence_decides_when_vocabulary_and_orientation_tie():
    tall = (100, 100, 120, 160)
    assert pick([
        tok("aaaa", 0.70, tall, 90),
        tok("bbbb", 0.95, tall, 270),
    ]) == "bbbb"


def test_near_tie_vocabulary_falls_through_to_orientation():
    """Vocabulary is bucketed to 1dp so tiny differences do not veto the other signals."""
    tall = (100, 100, 120, 160)
    a = tok("11.7", 0.60, tall, 90)
    b = tok("11.8", 0.99, tall, 0)
    # Same vocabulary bucket; a is orientation-plausible, b is not.
    assert reading_key(a)[0] == reading_key(b)[0]
    assert pick([a, b]) == "11.7"


# --- box shape and orientation rules --------------------------------------------------


def test_box_shape():
    assert box_shape((0, 0, 20, 60)) == "tall"
    assert box_shape((0, 0, 60, 20)) == "wide"
    assert box_shape((0, 0, 20, 21)) == "square"


def test_tall_box_orientation_rule():
    tall = (0, 0, 20, 60)
    assert orientation_is_plausible(tall, 90)
    assert orientation_is_plausible(tall, 270)
    assert not orientation_is_plausible(tall, 0)
    assert not orientation_is_plausible(tall, 180)


def test_wide_box_unconstrained_by_default():
    """Default rule is tall_only, because the wide mirror rule was wrong on real data."""
    wide = (0, 0, 60, 20)
    assert all(orientation_is_plausible(wide, a) for a in (0, 90, 180, 270))


def test_wide_box_constrained_under_both_rule():
    wide = (0, 0, 60, 20)
    assert orientation_is_plausible(wide, 0, rule="both")
    assert not orientation_is_plausible(wide, 90, rule="both")


def test_off_rule_disables_orientation_entirely():
    tall = (0, 0, 20, 60)
    assert all(orientation_is_plausible(tall, a, rule="off") for a in (0, 90, 180, 270))


def test_square_box_never_constrained():
    square = (0, 0, 20, 21)
    assert all(orientation_is_plausible(square, a, rule="both") for a in (0, 90, 180, 270))


# --- vocabulary scoring ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("good", "bad"),
    [
        ("OH", "HO"),
        ("MH", "HN"),
        ("KHH", "HHH"),
        ("TYOHUONE", "TrOHUONE"),
        ("9X21", "IZX6"),
        ("5515", "GISS"),
        ("11.7", "L'P!"),
        ("1049", "EFOL"),
    ],
)
def test_vocabulary_score_ranks_observed_pairs(good, bad):
    assert vocabulary_score(good) > vocabulary_score(bad)


# --- clustering behaviour -------------------------------------------------------------


def test_distinct_tokens_are_not_merged():
    kept = deduplicate([
        tok("MH", 0.9, (0, 0, 20, 30), 90),
        tok("11.7", 0.95, (200, 200, 240, 215), 90),
    ], iou_threshold=0.5)
    assert {t.text for t in kept} == {"MH", "11.7"}


def test_reading_order_is_top_to_bottom_then_left_to_right():
    kept = deduplicate([
        tok("lower", 0.9, (0, 500, 10, 510), 0),
        tok("upper_right", 0.9, (300, 10, 310, 20), 0),
        tok("upper_left", 0.9, (5, 10, 15, 20), 0),
    ], iou_threshold=0.5)
    assert [t.text for t in kept] == ["upper_left", "upper_right", "lower"]


def test_empty_input():
    assert deduplicate([], iou_threshold=0.5) == []


def test_every_cluster_yields_exactly_one_token():
    tokens = [
        tok("MH", 0.9, (100, 100, 120, 140), 180),
        tok("HN", 0.8, (101, 101, 119, 139), 270),
        tok("11.7", 0.95, (130, 100, 150, 140), 90),
        tok("L'P!", 0.5, (131, 101, 149, 139), 270),
    ]
    kept = deduplicate(tokens, iou_threshold=0.5)
    assert sorted(t.text for t in kept) == ["11.7", "MH"]
