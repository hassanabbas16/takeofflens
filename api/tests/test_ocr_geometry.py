"""Geometry tests for the multi-orientation merge.

These matter more than they look: if unrotate_point is wrong, OCR still "works" and the
viewer silently draws every box in the wrong place. The round-trip test pins it by checking
against cv2.rotate itself rather than against my own arithmetic.
"""

import cv2
import numpy as np
import pytest

from app.pipeline.ocr import (
    OcrToken,
    deduplicate,
    iou,
    rotate_image,
    unrotate_bbox,
    unrotate_point,
)

ANGLES = [0, 90, 180, 270]


@pytest.mark.parametrize("angle", ANGLES)
def test_unrotate_point_inverts_cv2_rotate(angle):
    """Mark one pixel, rotate, find it, and check unrotate_point maps it back."""
    height, width = 7, 11
    for y in (0, 3, height - 1):
        for x in (0, 5, width - 1):
            img = np.zeros((height, width), dtype=np.uint8)
            img[y, x] = 255

            rotated = rotate_image(img, angle)
            ys, xs = np.nonzero(rotated)
            assert len(xs) == 1, "rotation should preserve the single marked pixel"

            back_x, back_y = unrotate_point(float(xs[0]), float(ys[0]), angle, height, width)
            assert (round(back_x), round(back_y)) == (x, y)


@pytest.mark.parametrize("angle", ANGLES)
def test_rotated_image_shape(angle):
    img = np.zeros((7, 11, 3), dtype=np.uint8)
    rotated = rotate_image(img, angle)
    expected = (7, 11) if angle in (0, 180) else (11, 7)
    assert rotated.shape[:2] == expected


def test_rotate_image_rejects_unsupported_angle():
    with pytest.raises(ValueError, match="unsupported rotation"):
        rotate_image(np.zeros((4, 4, 3), dtype=np.uint8), 45)


def test_unrotate_point_rejects_unsupported_angle():
    with pytest.raises(ValueError, match="unsupported rotation"):
        unrotate_point(1.0, 1.0, 45, 10, 10)


def test_unrotate_bbox_round_trips_a_region():
    """A filled rectangle should come back to its original bounds through any rotation."""
    height, width = 40, 60
    x1, y1, x2, y2 = 10, 5, 25, 18
    img = np.zeros((height, width), dtype=np.uint8)
    img[y1:y2, x1:x2] = 255

    for angle in ANGLES:
        rotated = rotate_image(img, angle)
        ys, xs = np.nonzero(rotated)
        poly = np.array(
            [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]]
        )
        got = unrotate_bbox(poly, angle, height, width)
        # One pixel of slack: cv2.rotate is index-exact but max() is inclusive.
        assert abs(got[0] - x1) <= 1
        assert abs(got[1] - y1) <= 1
        assert abs(got[2] - (x2 - 1)) <= 1
        assert abs(got[3] - (y2 - 1)) <= 1


def test_unrotate_bbox_clamps_to_page():
    poly = np.array([[-5, -5], [1000, -5], [1000, 1000], [-5, 1000]])
    x1, y1, x2, y2 = unrotate_bbox(poly, 0, height=40, width=60)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (60, 40)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes sharing a 5x10 strip: inter 50, union 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_deduplicate_keeps_highest_confidence_reading():
    """The real case: 9X21 read correctly at one angle and mirrored at another."""
    good = OcrToken(text="9X21", confidence=0.789, bbox=(100, 100, 140, 120), angle=90)
    mirrored = OcrToken(text="IZX6", confidence=0.721, bbox=(101, 101, 139, 119), angle=270)

    kept = deduplicate([mirrored, good], iou_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].text == "9X21"


def test_deduplicate_keeps_distinct_tokens():
    a = OcrToken(text="MH", confidence=0.9, bbox=(0, 0, 20, 10), angle=0)
    b = OcrToken(text="11.7", confidence=0.95, bbox=(200, 200, 240, 215), angle=0)
    kept = deduplicate([a, b], iou_threshold=0.5)
    assert {t.text for t in kept} == {"MH", "11.7"}


def test_deduplicate_orders_top_to_bottom_then_left_to_right():
    lower = OcrToken(text="lower", confidence=0.9, bbox=(0, 500, 10, 510), angle=0)
    upper_right = OcrToken(text="upper_right", confidence=0.9, bbox=(300, 10, 310, 20), angle=0)
    upper_left = OcrToken(text="upper_left", confidence=0.9, bbox=(5, 10, 15, 20), angle=0)

    kept = deduplicate([lower, upper_right, upper_left], iou_threshold=0.5)

    assert [t.text for t in kept] == ["upper_left", "upper_right", "lower"]


def test_deduplicate_empty():
    assert deduplicate([], iou_threshold=0.5) == []


def test_cv2_rotate_matches_our_flag_table():
    """Guards against an OpenCV constant changing meaning under us."""
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert np.array_equal(rotate_image(img, 90), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    assert np.array_equal(rotate_image(img, 0), img)
