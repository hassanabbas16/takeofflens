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
