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


# --- combined label+area tokens and OCR near-misses -----------------------------------
# Every case below came from a real plan and cost a real area before it was fixed.


def test_combined_label_area_token_pairs_with_itself():
    """"khh 6.8" is one token. It must not be paired with another room's area."""
    from app.pipeline.pairing import TokenRef, find_candidate_pairs

    refs = [
        TokenRef(id=0, text="khh 6.8", confidence=0.9, bbox=(1710, 430, 1848, 474)),
        TokenRef(id=1, text="mh 11.5", confidence=0.9, bbox=(556, 435, 684, 480)),
    ]
    pairs = find_candidate_pairs(refs)
    assert len(pairs) == 2
    by_label = {p.label_text: p for p in pairs}
    assert by_label["khh"].area_m2 == 6.8
    assert by_label["khh"].combined is True
    assert by_label["mh"].area_m2 == 11.5


def test_combined_token_is_not_offered_as_another_labels_area():
    from app.pipeline.pairing import TokenRef, find_candidate_pairs

    refs = [
        TokenRef(id=0, text="khh 6.8", confidence=0.9, bbox=(100, 100, 200, 140)),
        TokenRef(id=1, text="OH", confidence=0.9, bbox=(210, 100, 240, 140)),
    ]
    pairs = find_candidate_pairs(refs)
    # OH has no area of its own, and must not steal 6.8 from khh.
    oh = [p for p in pairs if p.label_text == "OH" or p.label.text == "OH"]
    assert not oh or oh[0].area_m2 != 6.8


def test_rules_reports_the_label_alone_not_the_glued_token():
    """label_raw of "khh 6.8" would not match the reference label "KHH"."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="khh 6.8", confidence=0.9, bbox=(100, 100, 200, 140), angle=0),
    ])
    assert [r.label_raw for r in outcome.rooms] == ["khh"]
    assert outcome.rooms[0].area_m2 == 6.8
    assert outcome.rooms[0].room_type.value == "utility"


def test_multi_word_label_with_area():
    """OCR splits words: "autovaja 19.5" came back as "autova ja 19.5"."""
    from app.pipeline.parse_dims import DimKind, parse

    result = parse("autova ja 19.5")
    assert result.kind is DimKind.AREA
    assert result.area_m2 == 19.5


def test_single_character_ocr_slip_in_a_long_label_is_forgiven():
    """KEITTIO came back as KEITTLO - I misread as L."""
    from app.pipeline.parse_dims import DimKind, parse
    from app.pipeline.room_types import primary_type

    result = parse("keittlo 15.4")
    assert result.kind is DimKind.AREA
    assert result.area_m2 == 15.4
    assert primary_type("keittlo") == "kitchen"


def test_short_labels_are_never_fuzzy_matched():
    """K, H, S and WC are all real and one edit apart - guessing would corrupt them."""
    from app.pipeline.room_types import resolve_label

    assert resolve_label("K") == "K"
    assert resolve_label("H") == "H"
    # A two-character non-label must not snap to a real one.
    assert resolve_label("ZQ") is None
    assert resolve_label("XY") is None


def test_fuzzy_matching_does_not_invent_labels_from_noise():
    from app.pipeline.room_types import resolve_label

    assert resolve_label("QQQQQQ") is None
    assert resolve_label("123456") is None


def test_rules_approach_is_grounded_by_construction():
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="MH", confidence=0.95, bbox=(100, 100, 130, 130), angle=0),
        OcrToken(text="11.7", confidence=0.96, bbox=(100, 135, 140, 160), angle=0),
        OcrToken(text="9X21", confidence=0.88, bbox=(400, 400, 450, 425), angle=0),
    ])
    assert outcome.hallucinations == 0
    labels = {r.label_raw for r in outcome.rooms}
    assert "MH" in labels
    # The door code must never become a room.
    assert "9X21" not in labels


def test_rules_approach_costs_nothing():
    from app.pipeline.extract import extract_rules

    outcome = extract_rules([])
    assert outcome.usage.input_tokens == 0
    assert outcome.usage.output_tokens == 0
    assert outcome.source.value == "rules"


def test_rules_reports_labels_without_an_area_as_rooms_with_null_area():
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="OH", confidence=0.9, bbox=(100, 100, 130, 130), angle=0),
    ])
    assert len(outcome.rooms) == 1
    assert outcome.rooms[0].area_m2 is None


def test_soft_grounding_issues_are_not_counted_as_hallucinations():
    """The vlm path reads areas OCR missed; that is not fabrication."""
    from app.pipeline.grounding import check_grounding
    from app.pipeline.pairing import TokenRef
    from app.schemas import ExtractedRoom, PlanExtraction, RoomType

    refs = [TokenRef(id=0, text="MH", confidence=0.9, bbox=(0, 0, 20, 20))]
    extraction = PlanExtraction(
        rooms=[ExtractedRoom(
            label_raw="MH", room_type=RoomType("bedroom"), area_m2=11.7,
            width_m=None, length_m=None, source_token_ids=None, confidence=0.9,
        )],
        notes=None,
    )
    soft = check_grounding(extraction, refs, hard=False)
    assert soft.hallucination_count == 0
    assert soft.soft_signal_count == 1

    hard = check_grounding(extraction, refs, hard=True)
    assert hard.hallucination_count > 0


def test_one_area_is_not_shared_between_several_rooms():
    """Observed live: four rooms all got area 16.0 from one stray token."""
    from app.pipeline.pairing import TokenRef, find_candidate_pairs

    area = TokenRef(id=99, text="16.0", confidence=0.9, bbox=(300, 700, 360, 760))
    refs = [
        TokenRef(id=0, text="KH", confidence=0.9, bbox=(334, 591, 364, 630)),
        TokenRef(id=1, text="WC", confidence=0.9, bbox=(335, 654, 362, 690)),
        TokenRef(id=2, text="ET", confidence=0.9, bbox=(334, 793, 372, 830)),
        area,
    ]
    pairs = find_candidate_pairs(refs)
    claimed = [p for p in pairs if p.area.id == 99]
    assert len(claimed) == 1, [p.label_text for p in claimed]


def test_the_closest_label_wins_a_contested_area():
    from app.pipeline.pairing import TokenRef, find_candidate_pairs

    refs = [
        TokenRef(id=0, text="KH", confidence=0.9, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text="WC", confidence=0.9, bbox=(100, 400, 130, 430)),
        TokenRef(id=2, text="16.0", confidence=0.9, bbox=(100, 135, 150, 160)),
    ]
    pairs = [p for p in find_candidate_pairs(refs) if p.area.id == 2]
    assert len(pairs) == 1
    assert pairs[0].label_text == "KH"


def test_a_label_losing_a_contested_area_still_appears_as_a_room():
    """Losing the contest means no area, not disappearing."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="KH", confidence=0.9, bbox=(100, 100, 130, 130), angle=0),
        OcrToken(text="WC", confidence=0.9, bbox=(100, 180, 130, 210), angle=0),
        OcrToken(text="16.0", confidence=0.9, bbox=(100, 135, 150, 160), angle=0),
    ])
    by_label = {r.label_raw: r.area_m2 for r in outcome.rooms}
    assert set(by_label) == {"KH", "WC"}
    assert sorted(v is None for v in by_label.values()) == [False, True]


def test_single_letter_plus_digit_is_not_a_room():
    """"H7" came out of OCR noise and became a "room" on a real upload."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="H7", confidence=0.65, bbox=(100, 100, 130, 130), angle=0),
    ])
    assert outcome.rooms == []


# --- area corroboration ----------------------------------------------------------------
#
# Real regressions from the 6-plan hand-verified set (eval/ground_truth/printed_areas_6.json),
# where all 29 printed areas are decimals and four of the six plans print no area at all.


def test_bare_integer_is_not_an_area_when_the_page_prints_no_real_one():
    """Plan 2536: OCR shattered the apartment summary "4H K KH WC 90 M2" into fragments,
    and "016" and "2" landed under WC and ET and were reported as 16.0 and 2.0 m2. The page
    prints no areas at all, so nothing corroborates a bare integer."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="WC", confidence=0.84, bbox=(335, 654, 362, 701), angle=90),
        OcrToken(text="016", confidence=0.78, bbox=(340, 719, 359, 759), angle=0),
        OcrToken(text="ET", confidence=0.99, bbox=(334, 856, 362, 900), angle=270),
        OcrToken(text="2", confidence=0.99, bbox=(350, 793, 372, 810), angle=270),
    ])
    assert {r.label_raw for r in outcome.rooms} == {"WC", "ET"}
    assert all(r.area_m2 is None for r in outcome.rooms)


def test_bare_integer_is_accepted_when_a_sibling_area_corroborates_it():
    """The rule is corroboration, not a ban: a page that demonstrably prints areas as
    decimals may also print an integer one, and that one is kept."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="MH", confidence=0.99, bbox=(100, 100, 130, 130), angle=0),
        OcrToken(text="11.7", confidence=0.95, bbox=(100, 135, 150, 160), angle=0),
        OcrToken(text="OH", confidence=0.99, bbox=(300, 100, 330, 130), angle=0),
        OcrToken(text="24", confidence=0.95, bbox=(300, 135, 340, 160), angle=0),
    ])
    by_label = {r.label_raw: r.area_m2 for r in outcome.rooms}
    assert by_label["MH"] == pytest.approx(11.7)
    assert by_label["OH"] == pytest.approx(24.0)


def test_decimal_areas_survive_the_corroboration_rule():
    """Guard against over-correcting: the common printed form must be untouched."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="KHH", confidence=0.99, bbox=(100, 100, 140, 130), angle=0),
        OcrToken(text="10.8", confidence=0.95, bbox=(100, 135, 150, 160), angle=0),
    ])
    assert [(r.label_raw, r.area_m2) for r in outcome.rooms] == [("KHH", pytest.approx(10.8))]


def test_area_beside_the_apartment_type_code_is_not_a_room_area():
    """Plan 2504: "3H+KT+S 61,0 m2" is the whole unit. OCR split the code from its area and
    read it as "BH+KT+$", so the two are tied together geometrically, not lexically. The
    61 m2 was being handed to the living room printed just below it."""
    from app.pipeline.extract import extract_rules
    from app.pipeline.ocr import OcrToken

    outcome = extract_rules([
        OcrToken(text="BH+KT+$", confidence=0.79, bbox=(434, 608, 489, 774), angle=0),
        OcrToken(text="61,0m2", confidence=0.77, bbox=(396, 611, 441, 731), angle=0),
        OcrToken(text="OH", confidence=0.99, bbox=(425, 876, 465, 932), angle=0),
    ])
    assert [r.area_m2 for r in outcome.rooms] == [None]


def test_apartment_type_code_detection_tolerates_ocr_mangling():
    from app.pipeline.parse_dims import looks_like_apartment_code

    assert looks_like_apartment_code("3H+KT+S")
    assert looks_like_apartment_code("2H+KK")
    assert looks_like_apartment_code("BH+KT+$")     # 3 read as B, S as $
    assert not looks_like_apartment_code("MH")
    assert not looks_like_apartment_code("11.7")
    assert not looks_like_apartment_code("OLESKELU+RUOK")  # compound room label, not a code
