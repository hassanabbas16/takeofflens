"""Four-way classification of an extracted area against room-level ground truth.

Shared by the SVG-referenced scorer (all 50 plans, automatic ground truth) and the gold-set
scorer (5 plans, hand-verified), so both report the same categories and a number from one can
be read against the other.

Why four categories and not a single "accuracy"
-----------------------------------------------
The failures are different defects with different fixes, and a single score hides which one
is happening:

``correct``
    right room, right area.

``wrong_value``
    the room is real and was found, but the number attached to it is wrong. An extraction
    problem.

``misattributed``
    the number is a real area on this page, put against the wrong room. A *pairing* problem -
    the page was read correctly and the assignment failed.

``hallucinated``
    neither the room nor the number corresponds to anything on the page.

Matching areas as a bare multiset - which is what the Tier 3 results table did - cannot tell
these apart, and worse, scores wrong answers as right: on plan 416 it matched a `K 7.0`
(the kitchen is really 11.6) against an unrelated 6.8 m2 room and counted it correct. Pairing
on the room label first is what stops that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

sys.path.insert(0, "/app")

from app.pipeline.room_types import label_key


def close(a: float, b: float, tolerance: float) -> bool:
    """Relative comparison, taken against the ground-truth value."""
    return abs(a - b) <= tolerance * max(abs(b), 1e-6)


@dataclass
class AreaScore:
    correct: int = 0
    # Signed relative offsets (predicted - gold) / gold for every label-matched pair,
    # whether or not it fell inside tolerance. This is how the polygon-vs-printed
    # convention gap gets measured instead of guessed.
    offsets: list[float] = field(default_factory=list)
    wrong_value: int = 0
    misattributed: int = 0
    hallucinated: int = 0
    # Ground-truth rooms with no prediction matched to them.
    missed: int = 0
    gold_rooms: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def reported(self) -> int:
        return self.correct + self.wrong_value + self.misattributed + self.hallucinated

    @property
    def precision(self) -> float:
        return self.correct / self.reported if self.reported else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold_rooms if self.gold_rooms else 0.0

    def add(self, other: AreaScore) -> None:
        self.offsets.extend(other.offsets)
        self.correct += other.correct
        self.wrong_value += other.wrong_value
        self.misattributed += other.misattributed
        self.hallucinated += other.hallucinated
        self.missed += other.missed
        self.gold_rooms += other.gold_rooms


def score_plan(
    predicted: list[tuple[str, float]],
    gold: list[tuple[str, float]],
    tolerance: float,
    *,
    plan_id: str = "",
    collect: int = 0,
) -> AreaScore:
    """Classify every predicted (label, area) against the ground-truth rooms of one plan.

    Labels are compared through ``label_key`` so "khh 6.8" and "KHH" agree.

    Priority is deliberate, and the order is the argument:

    1. **correct** - claimed first, across all predictions, so a right answer is never lost
       to a wrong one that happened to be evaluated earlier.
    2. **misattributed** - the value matches an unclaimed room of a *different* name. The
       number is real; the assignment is not.
    3. **wrong_value** - a room of this name exists on the plan but no area matched. The
       room is right and the number is not.
    4. **hallucinated** - neither.

    A ground-truth room is consumed only by a correct or misattributed match, so repeating
    one area cannot inflate either. ``missed`` counts rooms nothing matched.
    """
    gold_norm = [(label_key(name or ""), area) for name, area in gold]
    pred_norm = [(label_key(name or ""), area) for name, area in predicted]

    remaining = list(gold_norm)
    result = AreaScore(gold_rooms=len(gold_norm))

    # Offsets are collected against the *nearest same-label* room, independently of the
    # claiming below, so the distribution is not truncated by the tolerance being measured.
    for label, value in pred_norm:
        same = [g for g in gold_norm if g[0] == label]
        if same:
            nearest = min(same, key=lambda g: abs(g[1] - value))
            if nearest[1] > 0:
                result.offsets.append((value - nearest[1]) / nearest[1])

    leftover: list[tuple[str, float]] = []
    for label, value in pred_norm:
        for i, (g_label, g_area) in enumerate(remaining):
            if label == g_label and close(value, g_area, tolerance):
                result.correct += 1
                remaining.pop(i)
                break
        else:
            leftover.append((label, value))

    labels_on_plan = {g_label for g_label, _ in gold_norm}
    for label, value in leftover:
        # Tested against the *whole* gold set and without consuming it: misattribution
        # counts predictions, not ground-truth slots. Two rooms both claiming one real area
        # are two misattributions - charging the second to "hallucinated" would say the
        # number was invented, which it was not. Only a correct match consumes a room, which
        # is what keeps recall honest.
        elsewhere = [
            (g_label, g_area)
            for g_label, g_area in gold_norm
            if close(value, g_area, tolerance) and g_label != label
        ]
        if elsewhere:
            result.misattributed += 1
            if collect and len(result.examples) < collect:
                g_label, g_area = elsewhere[0]
                result.examples.append(
                    f"{plan_id}: {label} {value:g} m2 is really {g_label}'s {g_area:.1f} m2"
                )
        else:
            if label in labels_on_plan:
                result.wrong_value += 1
                near = min(
                    (g for g in gold_norm if g[0] == label),
                    key=lambda g: abs(g[1] - value),
                    default=None,
                )
                if collect and near and len(result.examples) < collect:
                    result.examples.append(
                        f"{plan_id}: {label} {value:g} m2, polygon says {near[1]:.1f} m2 "
                        f"({100 * abs(near[1] - value) / max(near[1], 1e-9):.0f}% off)"
                    )
            else:
                result.hallucinated += 1
                if collect and len(result.examples) < collect:
                    result.examples.append(
                        f"{plan_id}: {label} {value:g} m2 - no such room, no such area"
                    )

    result.missed = len(remaining)
    return result
