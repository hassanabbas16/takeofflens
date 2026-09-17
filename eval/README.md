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
| 3 | `tier3_cost_probe.py` | **paid, gated** | — (runs all three Claude approaches) |
| 4 | `tier4_gold.py` + `label_helper.py` | free | hand-verified, 15 plans |

Tier 3 stops after 20 test plans and reports measured cost per page. The full run requires
explicit approval. See CLAUDE.md for the full tier definitions.

Every table in `results.md` states its sample size and whether its ground truth is automatic
(Tier 2) or hand-verified (Tier 4).

## Tier 4: labelling the gold set

Fifteen plans, hand-verified for the **areas the drawing actually prints**. This is the only
tier whose numbers may be described as hand-verified, and the only one that can support an
area-accuracy figure: Tier 2's areas come from the annotation polygon, not from the page.

The 15 are drawn from the 50-plan Tier 3 sample, so their results for all four approaches
are already paid for and scoring them costs nothing.

### 1. Select (already done, re-runnable)

```bash
docker compose run --rm --no-deps api python /eval/tier4_gold.py
```

Writes `eval/data/gold_15.txt` with the chosen plans, the reason each was chosen, and the
reason every other plan was not. Eligibility is >= 3 readable area tokens in the cached OCR;
the pick is round-robin across room-count bands, preferring plans whose area text OCR
garbled, because those are where the approaches actually differ.

### 2. Label

```bash
docker compose run --rm --no-deps api python /eval/label_helper.py
```

Needs a terminal, so run it exactly as above rather than through a pipe.

For each plan it writes `eval/labelling/<id>.png` - the page with every OCR token boxed and
numbered - and then walks the rooms `model.svg` lists. **Open that PNG beside the terminal.**
The ids in the prompt are the ids drawn on the image.

Each room shows the SVG label, the SVG polygon area (context only, not the answer), which
tokens on the page carry that label, and up to three candidate areas read from tokens near
it. Then:

| key | meaning |
| --- | --- |
| `Enter` or `y` | accept the suggested area as what the drawing prints |
| `2` / `3` | take the second or third suggestion instead |
| a number | the drawing prints this (`11.7` or `11,7` both work) |
| `n` | the drawing prints **no** area for this room - a real answer, not a skip |
| `l` | correct the label |
| `s` | skip, decide later (stays undecided) |
| `b` | back one room |
| `q` | save and quit |

Three things worth knowing while labelling:

- **The polygon area is not the answer.** It follows the inner wall face and disagrees with
  the printed figure by a percent or two, sometimes much more on L-shaped rooms. It is shown
  only so an implausible suggestion stands out.
- **`n` is a measurement.** Plenty of these drawings print no area for hallways, outdoor
  spaces and storage. Recording that is as valuable as recording a number, and it is what
  stops a fabricated area from being scored as correct.
- **"no token matches this label" usually means OCR read it rotated** - `OH` comes back as
  `HO`. The area is often still on the page; the tool lists the unclaimed ones so you can
  pick the value off the image and type it.

Progress is saved after every answer:

```bash
docker compose run --rm --no-deps api python /eval/label_helper.py --status
docker compose run --rm --no-deps api python /eval/label_helper.py --review 1041
```

Output goes to `eval/ground_truth/gold/<id>.json`, recording for each room how the answer was
reached (`confirmed_suggestion`, `typed`, `read_as_not_printed`), so a value accepted from the
machine's suggestion is distinguishable from one typed after reading the page.

### 3. Score, once the labelling is done

```bash
docker compose run --rm --no-deps api python /eval/tier4_area_accuracy.py
```

Free: `rules` is recomputed from the OCR cache and the three paid approaches are replayed
from the raw batch bodies in `eval/cache/batch_raw/`. Reports precision and recall per
approach, never averaged into one score - `rules` is conservative and `vlm` is liberal, and a
single figure would hide exactly that. Plans that are not finished are excluded and named, so
a partial labelling session gives a partial but honest table.
