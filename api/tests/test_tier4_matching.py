"""Tests for the Tier 4 room+area matcher.

The matcher lives in eval/, which is not importable from the api package, so it is loaded by
path. It is tested here rather than left untested because it is what turns a labelling
session into the project's headline accuracy number, and a silent bug in it would be
invisible - there is nothing to compare its output against.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EVAL = Path(__file__).resolve().parents[2] / "eval" / "tier4_area_accuracy.py"


def _load():
    """Import eval/tier4_area_accuracy.py without importing its heavy module-level deps."""
    spec = importlib.util.spec_from_file_location("tier4_area_accuracy", _EVAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tier4 = pytest.importorskip("cv2") and _load()
match = tier4.match
TOL = 0.05


def test_right_room_and_right_area_is_correct():
    result = match([("MH", 11.7)], [("MH", 11.7)], TOL)
    assert result == (1, 0, 0)


def test_right_area_on_the_wrong_room_is_misattributed_not_correct():
    """The number is really printed on the page; the assignment is what failed."""
    correct, mis, spurious = match([("OH", 11.7)], [("MH", 11.7)], TOL)
    assert (correct, mis, spurious) == (0, 1, 0)


def test_an_area_not_printed_anywhere_is_spurious():
    assert match([("MH", 99.9)], [("MH", 11.7)], TOL) == (0, 0, 1)


def test_a_label_correct_match_is_not_stolen_by_a_label_wrong_one():
    """Both rooms claim 11.7; MH owns it. Ordering must not decide the outcome.

    With a single greedy pass, OH (listed first) would take the only 11.7 and MH would then
    score as spurious - one correct match lost and two failures invented.
    """
    predicted = [("OH", 11.7), ("MH", 11.7)]
    gold = [("MH", 11.7)]
    assert match(predicted, gold, TOL) == (1, 1, 0)


def test_duplicate_labels_match_within_the_label():
    """Three MH rooms: nothing distinguishes one from another, so any pairing is allowed."""
    predicted = [("MH", 12.6), ("MH", 11.5)]
    gold = [("MH", 11.5), ("MH", 12.6)]
    assert match(predicted, gold, TOL) == (2, 0, 0)


def test_repeating_one_area_cannot_inflate_recall():
    predicted = [("MH", 11.7)] * 5
    correct, mis, spurious = match(predicted, [("MH", 11.7)], TOL)
    assert correct == 1
    assert mis == 0
    assert spurious == 4


def test_tolerance_is_relative_and_applied_to_the_gold_value():
    assert match([("MH", 11.9)], [("MH", 11.7)], TOL) == (1, 0, 0)   # 1.7% apart
    assert match([("MH", 12.6)], [("MH", 11.7)], TOL) == (0, 0, 1)   # 7.7% apart


def test_missing_predictions_lower_recall_without_inventing_failures():
    assert match([], [("MH", 11.7), ("OH", 22.6)], TOL) == (0, 0, 0)


def test_two_rooms_claiming_one_printed_area_are_both_misattributed():
    """Misattribution counts predictions, not gold slots.

    The tool put a real area against two wrong rooms; both are that same defect. Charging
    the second to "spurious" would say the number was invented, which it was not.
    """
    predicted = [("OH", 11.7), ("KH", 11.7)]
    assert match(predicted, [("MH", 11.7)], TOL) == (0, 2, 0)


def test_a_duplicate_of_an_already_matched_room_is_spurious_not_misattributed():
    """Same label, same area, reported twice: an extra unsupported claim, not a mis-assign."""
    assert match([("MH", 11.7), ("MH", 11.7)], [("MH", 11.7)], TOL) == (1, 0, 1)


def test_the_three_outcomes_partition_every_prediction():
    predicted = [("MH", 11.7), ("OH", 11.7), ("KH", 99.9), ("MH", 11.7)]
    gold = [("MH", 11.7)]
    correct, mis, spurious = match(predicted, gold, TOL)
    assert correct + mis + spurious == len(predicted)
