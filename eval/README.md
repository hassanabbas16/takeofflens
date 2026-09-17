# Evaluation

## Dataset: CubiCasa5K

5000 annotated Finnish floor plans. Not redistributed here — mounted read-only from
`DATASET_DIR`. `eval/data/` holds only plan **ID lists**, never image data.

- Source: CubiCasa5K, published by CubiCasa alongside the paper
  *"CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis"*
  (Kalervo et al., 2019).
- **License: to be confirmed before publishing this repo.** The dataset is distributed for
  research use; record the exact license from the official release page here, with the date
  checked, before the README cites any results from it. Do not assume.

### Splits
`train.txt` (4199), `val.txt` (399), `test.txt` (399). Each line is a plan directory, e.g.
`/high_quality_architectural/1191/`, containing `F1_original.png`, `F1_scaled.png`, `model.svg`.

### What the three folders actually contain

Measured by inspecting samples from each folder. Tier 1 quantifies this over the whole split.

| Folder | test split | Room labels | Areas / dimensions |
| --- | --- | --- | --- |
| `high_quality_architectural` | 270 | Finnish abbreviations | often, inconsistently |
| `high_quality` | 63 | Finnish abbreviations | rarely |
| `colorful` | 67 | none | none |

`colorful` plans are CubiCasa vector renders carrying only the boilerplate
"SUUNTAA-ANTAVA, EI MITTAKAAVASSA" and a logo. They are a negative set for OCR, not part of
the dimension evaluation.

### `model.svg` is the annotation, not the drawing
`F1_*.png` is the agent's plan; `model.svg` is CubiCasa's annotation of it. They do not carry
the same text, so SVG strings are never used as OCR ground truth. Dimension labels in the SVG
are `display: none` and are not rendered into the PNG.

Scale is a constant **100 SVG units = 1 metre** (median 100.02 over 5127 room edges across 200
test plans). Room area = shoelace area of the `Space` polygon / 10000.

## Tiers

| Tier | Script | Cost | Ground truth |
| --- | --- | --- | --- |
| 1 | `tier1_ocr_sweep.py` | free | — (OCR sweep, resumable + cached) |
| 2 | `tier2_ground_truth.py` | free | automatic, from `model.svg` |
| 2 | **`tier2_area_accuracy.py`** | free | **automatic — the primary area metric** |
| 3 | `tier3_cost_probe.py` | **paid, gated** | — (runs the three Claude approaches) |

Tier 3 stops after 20 test plans and reports measured cost per page. The full run requires
explicit approval. See CLAUDE.md for the full tier definitions.

Every table states its sample size and what its ground truth is. All of it is automatic:
nothing in the evaluation depends on a human labelling plans.

## How area accuracy is measured

### The method

Area accuracy is scored against the **shoelace area of the `model.svg` annotation polygon**,
for every named room of at least 1 m² on all 50 sampled plans — 691 rooms. It is free, it
covers every room rather than a sampled handful, and it is recomputed from cache, so
`eval/tier2_area_accuracy.py` costs nothing to re-run after a pipeline change.

A prediction is matched on **room and area together**. An area alone is not a takeoff: a tool
that reports the right number against the wrong room produces a quantity survey that does not
add up. Every prediction falls into exactly one of four buckets:

| Bucket | Meaning | What it points at |
| --- | --- | --- |
| `correct` | right room, area within tolerance | — |
| `wrong_value` | the room is real and was found, the number is wrong | extraction |
| `misattributed` | the number is a real area on the page, on the wrong room | **bbox pairing** |
| `hallucinated` | neither the room nor the number corresponds to anything | extraction |

Splitting `misattributed` out matters because it has a different fix from the others: the
page was read correctly and the assignment failed. Matching areas as a bare multiset — which
an earlier version did — cannot tell these apart, and worse, scores wrong answers as right:
on plan 416 it matched a `K 7.0` (the kitchen is really 11.6) against an unrelated 6.8 m²
room and counted it correct.

### The limitation, and why 10% is reported beside 5%

**A polygon area is not the figure printed on the drawing.** The polygon follows the inner
wall face; the printed figure uses the estate agent's own convention. Measured over 145
label-matched pairs on these plans, the extracted value sits a median **+4.4%** from its
polygon, and only **30%** of pairs land within 5%.

On plan 416, where every extracted value was checked against the drawing by eye, the
*correct* readings sit +8% to +13% from their polygons — MH 12.3 vs 11.24, TH 7.4 vs 6.83,
KHH 6.0 vs 5.46. A 5% band cannot contain a systematic offset that size.

The consequence is concrete and auditable. The OCR `m²` repair is a known-good change that
took plan 416 from 3 of 6 correct to 8 of 9:

| Tolerance | `rules` correct on 416, before | after |
| --- | --- | --- |
| 5% | 0 | 0 |
| 10% | 2 | 6 |

**At 5% the metric cannot see a real improvement; at 10% it can.** That is a property of the
ground truth, not of the matcher. So the results table reports both, and the 10% row is the
one that tracks the pipeline. Neither is an absolute accuracy figure against the page —
they are comparative numbers between approaches, which is what they are good for.

## Optional: reading a plan by hand

`eval/label_helper.py` walks a plan room by room and records what the drawing prints, to
`eval/ground_truth/manual/<id>.json`. **No report reads it.** It exists because reading a
plan by hand is the fastest way to understand a disagreement between an approach and the
polygons — a debugging aid, not a source of metrics.

```bash
docker compose run --rm --no-deps api python /eval/label_helper.py 1191   # needs a TTY
```

It writes `eval/labelling/<id>.png` with every OCR token boxed and numbered; open that
beside the terminal, since the ids in the prompt are the ids drawn on the image. `Enter`
accepts the suggested area, a number overrides it, `n` records that the drawing prints none,
`s` skips, `b` goes back, `q` saves and quits. `--status` shows what has been recorded and
`--review <id>` reopens a plan.
