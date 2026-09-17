# Area accuracy vs the `model.svg` polygons - all 50 sampled plans

- Ground truth: **automatic (Tier 2)**, the shoelace area of the CubiCasa annotation polygon. Not hand-verified.
- 45 plans, 691 annotated rooms of at least 1 m2 with a name
- Match tolerance: **5%**, on room **and** area together
- Computed entirely from cache. No API calls.

> **What a polygon area is and is not.** It is the area CubiCasa's annotator drew,
> following the inner wall face. It is *not* the figure printed on the drawing,
> which uses the agent's own convention. The two disagree by a few percent, so a
> correct reading can still be scored wrong here. How often that happens is measured
> on a 5-plan hand-labelled set - see `tier4_gold_validation.md` - rather than
> assumed. Treat these as comparative numbers between approaches, which is what they
> are good for, not as absolute accuracy against the page.

## The polygon is systematically smaller than the printed figure

Measured over every label-matched `rules` pair on these plans (n = 145), as (extracted - polygon) / polygon:

- median offset **+4.4%**
- within 5% of the polygon: **30%** of pairs
- within 10%: **48%** of pairs

The polygon follows the inner wall face; the printed figure does not. On plan 416, where every extracted value was checked against the drawing by eye, the correct readings sit +8% to +13% from their polygons - MH 12.3 vs 11.24, TH 7.4 vs 6.83, KHH 6.0 vs 5.46. **A 5% band cannot contain a systematic offset of that size**, so at 5% this metric scores correct readings as wrong.

That is a property of the ground truth, not of the matcher, which is why both tolerances are reported. Quantifying it against the drawing is the one job the 5-plan hand-labelled set exists to do.

## Does this metric track a real improvement?

The OCR area-unit repair is a known-good change. On plan 416, every extracted value was checked against the drawing by eye: the repair took it from 3 of 6 correct to 8 of 9, and corrected two already-wrong values. `rules` correct on that plan, before and after the repair:

| Tolerance | Before | After |
| --- | --- | --- |
| 5% | 0 | 0 |
| 10% | 2 | 6 |

**At 5% the metric cannot see the improvement at all; at 10% it does.** That is the convention offset, not the matcher: room pairing is working, and it is what stops the older area-only matcher from scoring a wrong kitchen value as correct by accidentally matching it against an unrelated room. The 10% row is the one that reflects the pipeline.

## Results at 5%

| Approach | Precision | Recall | Correct | Wrong value | Misattributed | Hallucinated | Reported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rules` | 24% | 6% | 43 | 66 | 51 | 20 | 180 |
| `ocr+llm` | 23% | 6% | 39 | 60 | 50 | 20 | 169 |
| `vlm` | 20% | 5% | 37 | 46 | 75 | 25 | 183 |
| `hybrid` | 26% | 8% | 52 | 63 | 65 | 23 | 203 |

## Results at 10%

The tolerance that actually accommodates the +4% convention offset:

| Approach | Precision | Recall | Correct | Wrong value | Misattributed | Hallucinated | Reported |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rules` | 38% | 10% | 69 | 43 | 54 | 14 | 180 |
| `ocr+llm` | 38% | 9% | 64 | 34 | 60 | 11 | 169 |
| `vlm` | 36% | 10% | 66 | 24 | 82 | 11 | 183 |
| `hybrid` | 42% | 12% | 85 | 34 | 70 | 14 | 203 |

**Recall** is against all 691 annotated rooms. That denominator is a floor, not a target: many of those rooms have no area printed on the drawing at all, so no approach could reach 100% by reading the page correctly. It is comparable between approaches, which is the point.

## Tolerance sensitivity

5% is a choice. Correct counts at other tolerances:

| Tolerance | `rules` | `ocr+llm` | `vlm` | `hybrid` |
| --- | --- | --- | --- | --- |
| 2% | 14 | 14 | 14 | 18 |
| 5% | 43 | 39 | 37 | 52 |
| 10% | 69 | 64 | 66 | 85 |
| 20% | 81 | 77 | 87 | 107 |

The gap between 5% and 10% is mostly the polygon-vs-printed convention difference, not extraction error - which is the limitation the gold set exists to quantify.

## Failure examples

**`rules`**

- 366: TERASSI 276 m2, polygon says 108.1 m2 (155% off)
- 366: OH 71.5 m2, polygon says 45.9 m2 (56% off)
- 366: PUKUH 10 m2, polygon says 9.4 m2 (7% off)
- 416: VAR 6.7 m2 is really TH's 6.8 m2
- 416: KHH 6 m2 is really VAR's 6.2 m2
- 416: MH 12.3 m2, polygon says 11.2 m2 (9% off)

**`ocr+llm`**

- 366: TERASSI 276 m2, polygon says 108.1 m2 (155% off)
- 366: OH 71.5 m2, polygon says 45.9 m2 (56% off)
- 366: PUKUH 10 m2, polygon says 9.4 m2 (7% off)
- 416: VAR 6.7 m2 is really TH's 6.8 m2
- 416: KHH 6 m2 is really VAR's 6.2 m2
- 416: MH 12.3 m2, polygon says 11.2 m2 (9% off)

**`vlm`**

- 366: KH 5 m2 - no such room, no such area
- 366: OH 18.5 m2, polygon says 45.9 m2 (60% off)
- 366: K 12.1 m2 is really MH's 12.6 m2
- 416: VAR 6.7 m2 is really TH's 6.8 m2
- 416: MH 12.3 m2, polygon says 11.2 m2 (9% off)
- 416: WC 2.3 m2, polygon says 2.0 m2 (17% off)

**`hybrid`**

- 366: TERASSI 276 m2, polygon says 108.1 m2 (155% off)
- 366: OH 71.5 m2, polygon says 45.9 m2 (56% off)
- 366: PUKUH 10 m2, polygon says 9.4 m2 (7% off)
- 416: VAR 6.7 m2 is really TH's 6.8 m2
- 416: MH 12.3 m2, polygon says 11.2 m2 (9% off)
- 416: KHH 6 m2 is really VAR's 6.2 m2

Skipped 5 plans (no OCR cache or unusable SVG scale): 1041, 1339, 7691, 20000, 20003
