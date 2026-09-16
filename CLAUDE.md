# TakeoffLens — Build Spec

## What we're building
A blueprint analysis tool. User uploads an architectural floor plan (PDF or image). The system extracts text (room labels, dimensions, annotations) with OCR, parses it into structured takeoff data (rooms, dimensions, areas, counts), and shows the results in a web viewer with bounding-box overlays and CSV/JSON export.

This is a portfolio project for an AI/CV full-stack role. It must actually work end-to-end, be runnable with one command, and include an honest evaluation with real numbers. Never fabricate metrics or fake results.

## Stack (do not substitute without asking)
- Backend: Python 3.11, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2
- CV/OCR: OpenCV, PaddleOCR (EasyOCR as a documented fallback if Paddle install fails), PyMuPDF for PDF -> image
- LLM/VLM: Anthropic Claude API via the official `anthropic` Python SDK
  (switched from OpenAI on request, 2026-09-17)
  - Key from env `ANTHROPIC_API_KEY`
  - Model names from env: `ANTHROPIC_TEXT_MODEL` and `ANTHROPIC_VISION_MODEL` (never
    hardcode; check the current Anthropic docs and put sensible defaults in `.env.example`)
  - Default `claude-haiku-4-5` for both: cheapest current model ($1/$5 per MTok) and
    vision-capable, so one model serves all three approaches. `claude-sonnet-5` ($2/$10) is
    the step up if recall is insufficient.
  - Use Structured Outputs via `client.messages.parse(output_format=<Pydantic model>)`,
    which returns a validated instance on `response.parsed_output`
  - `max_tokens` is required by the Messages API. Hitting it truncates the extraction, so a
    `max_tokens` stop reason is treated as a failure, not as a usable result.
  - Temperature 0 where the model accepts it. Sonnet 5 and Opus 5 removed the sampling
    parameters and reject them; that is detected at runtime, not hardcoded.
- DB: PostgreSQL 16
- Frontend: Next.js (App Router), TypeScript, Tailwind CSS
- Infra: Docker + docker-compose (api, web, db)
- Dataset: CubiCasa5K, mounted read-only from host via `DATASET_DIR` (never copied into the repo)
- Tests: pytest (backend)

## Dataset (CubiCasa5K)

Mounted read-only, never vendored into the repo.

- Host path comes from env `DATASET_DIR` (plans) and `DATASET_COCO_DIR` (COCO annotations).
- docker-compose mounts them read-only at `/data/cubicasa5k` and `/data/cubicasa5k_coco` inside `api`.
- Put the host paths in `.env` only; `.env.example` documents them with placeholder values.
- `eval/data/` is NOT a copy of the dataset. It holds only the selected plan ID lists
  (`eval/data/*.txt`) that point into the mounted dataset.

### Layout
```
$DATASET_DIR/
  train.txt val.txt test.txt     # one plan dir per line, e.g. /high_quality_architectural/1191/
  colorful/<id>/                 # F1_original.png, F1_scaled.png, model.svg
  high_quality/<id>/
  high_quality_architectural/<id>/
$DATASET_COCO_DIR/
  train_coco_pt.json val_coco_pt.json test_coco_pt.json
```
Split sizes: train 4199, val 399, test 399 plan dirs.

### Text characteristics — measured, and they drive the eval tiers
The three folders are very different documents, and only one of them has numbers on it.
Verified by inspecting sample plans from each folder; Tier 1 exists to quantify this over the
whole test split rather than trusting the samples.

| Folder | test split | Room labels on the image | Areas / dimensions on the image |
| --- | --- | --- | --- |
| `high_quality_architectural` | 270 | yes, Finnish abbreviations | often, but inconsistent |
| `high_quality` | 63 | yes, Finnish abbreviations | rarely |
| `colorful` | 67 | none — CubiCasa vector renders | none |

- `colorful` pages carry only boilerplate ("SUUNTAA-ANTAVA, EI MITTAKAAVASSA") and the CubiCasa
  logo. They are a legitimate *negative* set for OCR, not part of the dimension eval.
- Room text is frequently rotated 90 degrees. The OCR angle classifier is mandatory, not optional.
- Scanned pages carry handwriting, stamps and pen marks.

### `model.svg` is the annotation, not the drawing
`F1_*.png` is the real estate agent's plan; `model.svg` is CubiCasa's hand annotation of it.
They do not contain the same text, so never treat SVG strings as OCR ground truth.

- Room polygons live under `class="Space <Type>"`; `<Type>` is already English
  (`Bedroom`, `Kitchen`, `Bath Shower`, `Outdoor Balcony`, `Undefined`, ...).
- `TextLabel NameLabel` holds the Finnish label (`MH`, `OH`, `KHH`). XML entities appear
  here (`KEITTI&#xD6;` = `KEITTIÖ`) and must be unescaped.
- `TextLabel DimensionMeasureLabel` holds `12'3" x 6'1"` and `3.63 m x 3.90 m`, but it is
  `style="display: none"` on every plan sampled, so those strings are **not** rendered into the
  PNG and are not what OCR sees.
- **Scale is a constant 100 SVG units = 1 metre.** Measured over 5127 room edges across 200 test
  plans: median ratio 100.02. The ~7% of edges that deviate are L-shaped rooms whose bounding box
  is not the nominal labelled dimension, not a change in scale. The ground-truth generator must
  still compute the per-plan median ratio and flag any plan that deviates from 100.
- Room area is therefore the shoelace area of the `Space` polygon / 10000.
  Spot-checked against the areas printed on the drawing: agreement within ~1-2%
  (polygon follows the inner wall face; the printed figure uses the agent's own convention).
  Close enough to pre-fill a labelling helper, not close enough to be an oracle — which is
  exactly why Tier 4 is hand-verified.

## Repo structure
```
takeofflens/
  api/
    app/
      main.py
      config.py            # pydantic-settings, env vars
      db.py
      models.py            # SQLAlchemy models
      schemas.py           # Pydantic schemas (also the model's output_format schema)
      llm.py               # thin Anthropic client wrapper: retries, timeouts, token/cost logging
      routes/
        plans.py
      pipeline/
        ingest.py          # PDF/image -> normalized page images
        preprocess.py      # OpenCV steps
        ocr.py             # OCR wrapper, returns text + bbox + confidence
        parse_dims.py      # regex dimension parser -> meters + area (Finnish formats)
        room_types.py      # Finnish -> English room type mapping
        classify.py        # text model: OCR tokens -> structured rooms JSON
        vlm_direct.py      # vision model baseline (image -> rooms JSON)
        run.py             # orchestrates the pipeline
    alembic/
    tests/
    Dockerfile
    requirements.txt
  web/
    app/
      page.tsx             # upload
      plans/[id]/page.tsx  # viewer
    components/
      PlanCanvas.tsx       # image + bbox overlay, hover to highlight
      RoomsTable.tsx
    Dockerfile
  eval/
    data/                  # plan ID lists only (gold_15.txt) - never image data
    cache/                 # ocr/ and llm/ per-plan caches (gitignored)
    ground_truth/
      auto/                # Tier 2, generated from model.svg
      gold/                # Tier 4, hand-verified
    results/               # per-tier reports
    tier1_ocr_sweep.py
    tier2_ground_truth.py
    tier3_llm.py
    tier4_gold.py
    label_helper.py
    pricing.yaml           # model rates + date taken
    run_eval.py            # computes metrics, writes eval/results.md
    README.md              # dataset source, license, tier descriptions
  docker-compose.yml
  .env.example
  README.md
```

## Data model
- `plans`: id (uuid), filename, status (pending|processing|done|failed), error, created_at
- `pages`: id, plan_id, page_number, image_path, width, height
- `ocr_tokens`: id, page_id, text, confidence, bbox (x1,y1,x2,y2 as ints)
- `rooms`: id, page_id, name, room_type, width_m, length_m, area_m2, source ("ocr+llm" | "vlm" | "hybrid" | "detector"), raw_text, bbox (nullable), confidence, source_token_ids, grounded
  - **`area_m2` is the primary extracted field.** Finnish plans print a single area under the
    room label (`MH 11.7`), so area is what is actually on the page.
  - `width_m` and `length_m` are **nullable and never derived**. They are populated only when
    the page genuinely prints a `width x length` pair. Never back out a width and length from
    an area (no square roots, no assumed aspect ratio), and never infer an area from a
    dimension pair that was not printed. A null here is a real measurement, not a gap to fill.
- `llm_calls`: id, page_id, purpose, model, input_tokens, output_tokens, latency_ms, created_at (for cost/latency reporting)

## Pipeline requirements
1. **Ingest**: PDF pages rendered at 300 DPI via PyMuPDF; images passed through. Store page images to a local `storage/` volume behind a `Storage` interface so S3 can be swapped in later.
2. **Preprocess** (each step toggleable via config, for eval ablations): grayscale, denoise, adaptive threshold, deskew, upscale if the short side < 2000px.
3. **OCR**: return a list of `{text, confidence, bbox}`. Handle rotated text (PaddleOCR angle classifier on). Drop tokens below a configurable confidence.
4. **Dimension parsing**: regex + unit normalization. Convert all to metres, compute area.
   Unit tests for every format plus malformed input.

   Generic formats (keep — the parser should not be dataset-specific):
   `12'6" x 10'`, `12' - 6" x 10' - 0"`, `3.5 x 4.2`, `3.5m x 4.2m`, `3500 x 4200` (mm).

   Finnish / CubiCasa formats, which are what this dataset actually prints:
   - **Comma decimals**: `12,5` means 12.5. Accept comma and period as decimal separators.
     Note: the architectural plans sampled so far printed a period (`MH 11.7`); Finnish
     convention is the comma. Support both and let Tier 1 report which dominates — do not
     assume either.
   - **Bare area under a room label**, which is the dominant form here: the label is
     `MH` with `11.7` beneath it, meaning a 11.7 m² bedroom. This is a single **area**,
     not a `width x length` pair. `area_m2` is set and `width_m`/`length_m` stay null.
     Do not fabricate a width and length by taking a square root.
   - Area units appear as `m²`, `m2` and `M2`.
   - **Apartment summary strings**: `4H K KH WC 90 M2` (4 rooms + kitchen + bath + WC, 90 m²)
     describe the whole unit, not a room. Parse to a separate summary, never as a room.
   - **Wall run dimensions in mm** along the page margin: `19940`, `11000`, `3970`. Bare
     4-5 digit integers. These are building dimensions, not room dimensions.
   - **Door and window size codes**: `9X21`, `14X12`, `8X21`, `12X16`. These are Finnish
     door/window decimetre codes and are extremely common on architectural plans. They look
     exactly like a `w x h` dimension pair and are the parser's main false-positive source.
     **They must be classified as door/window codes and excluded from room dimensions.**
     Heuristic: both operands are bare integers <= 30 with no unit, written with `X`/`x`
     and no decimal separator. Unit-test this against real room dimensions.
   - **Elevation marks**: `+46.290`, `+46.950`. Never a dimension.

   `parse_dims.py` returns a typed result carrying which format matched, so eval can report
   accuracy per format rather than one aggregate number.
5. **Classify**: send OCR tokens (text + bbox) to the text model with a strict JSON schema. Associate each room label with its nearest dimension string using bbox proximity *before* the LLM call, and pass candidate pairs in the prompt. Validate with Pydantic; retry once on failure; never crash the job. Temperature 0.
6. **VLM baseline**: send the page image (base64, downscaled to a sensible max size to control cost) directly to the vision model, same output schema. Store with `source="vlm"`.
6b. **Hybrid**: page image *plus* the OCR token list as hints, to the vision model, citing
   token ids where it uses them and `null` where it read something OCR missed. Store with
   `source="hybrid"`. Three approaches, one schema, so eval compares the evidence rather
   than three different pipelines.
6c. **Grounding check.** For `ocr+llm` and `hybrid` this is hard: every cited token id must
   exist on the page, and every non-null `area_m2` must match an area the parser actually
   found in a token. Failures are counted as hallucinations and the room is kept with
   `grounded=False` so it stays visible and countable. For `vlm` the same checks run as a
   **soft** signal only - the vision model can legitimately read an area OCR missed, and on
   these plans OCR misses about half of them, so failing a VLM room for disagreeing with
   OCR would measure OCR rather than the model.
7. Log every model call to `llm_calls` (tokens, latency). Anthropic reports cache reads and
   cache writes as counters **separate from** `input_tokens`, so all four are stored and
   summed independently when costing - netting them would undercount. Cost per page is computed in eval from a pricing config file, not hardcoded in logic.
8. Processing runs as a FastAPI BackgroundTask for now. Keep the interface clean enough to move to a queue later; note that in the README.

## Finnish room labels

`api/app/pipeline/room_types.py` holds the Finnish -> English mapping, as data, not inline
literals. The observed label frequencies below are counted over 120 test-split `model.svg`
files, so the mapping covers the common cases first.

| Finnish | English `room_type` | seen |
| --- | --- | --- |
| MH (makuuhuone) | bedroom | 204 |
| ULKOTILA | outdoor | 131 |
| OH (olohuone) | living_room | 99 |
| WC | wc | 94 |
| ET (eteinen) | entry | 90 |
| K / KEITTIÖ / KK (keittokomero) | kitchen | 105 |
| VH (vaatehuone) | walk_in_closet | 63 |
| TK (tuulikaappi) | draught_lobby | 48 |
| PH / PESUH / PSH (pesuhuone) | washroom | 65 |
| VAR / VARASTO | storage | 54 |
| H (huone) | room | 35 |
| KHH (kodinhoitohuone) | utility | 33 |
| PARVEKE / PARV. | balcony | 31 |
| KH / KPH (kylpyhuone) | bathroom | 40 |
| TERASSI | terrace | 25 |
| KUISTI | porch | 17 |
| AULA | hall | 10 |
| AH (askarteluhuone) | hobby_room | 10 |
| RUOK (ruokailu) | dining | 9 |
| AUTOKATOS | carport | 9 |
| AUTOTALLI | garage | 7 |
| TEKN | technical_room | 8 |
| ULLAKKO | attic | 7 |
| SH (saunahuone) / SAUNA | sauna | 7 |
| KÄYTÄVÄ | corridor | 6 |
| PUKUH (pukuhuone) | dressing_room | 5 |
| TYÖHUONE / TH | office | 7 |
| ALKOVI | alcove | 4 |
| UNDEFINED | undefined | 243 |

Rules:
- Match case-insensitively, strip trailing `.`, and unescape XML entities before lookup.
- Compound labels joined with `+` (`OLESKELU+RUOK`) map to a list of types; keep the
  raw text in `raw_text` and pick the first as the primary `room_type`.
- Unknown labels get `room_type="unknown"` with `raw_text` preserved. Never guess.
- `UNDEFINED` is the single most common label in the dataset. It is a real annotation value
  meaning the annotator did not assign a type — map it to `undefined`, and exclude it from
  room-type accuracy metrics, reporting its share separately.
- The mapping is also used to score the LLM output, so it must be a pure lookup with no
  LLM involvement.

## API
- `POST /plans` (multipart upload) -> `{id, status}`
- `GET /plans/{id}` -> status + pages + rooms
- `GET /plans/{id}/pages/{n}/image`
- `GET /plans/{id}/export?format=csv|json&source=ocr+llm|vlm`
- `GET /health`
- CORS for the web container. Max upload 20MB. Allow only pdf/png/jpg.

## Frontend
- Upload page with drag-and-drop and status polling.
- Viewer: plan image on the left with OCR bbox overlays (toggle on/off), rooms table on the right. Hovering a row highlights its bbox. Toggle between "OCR+LLM" and "VLM" results. Export buttons.
- Clean, minimal Tailwind UI. No component libraries needed.

## Evaluation — tiered

Cost and hand-labelling effort both scale with plan count, so the eval is staged. Each tier is a
separate entry point under `eval/`, each writes machine-readable output that the next tier reads,
and nothing re-does work a previous tier already did.

All tiers read plans from the mounted `DATASET_DIR`. None of them copy image data into the repo.

### Tier 1 — OCR sweep over all plans (no LLM, no cost)
`eval/tier1_ocr_sweep.py`

- Runs ingest + preprocess + OCR over every plan in the split(s) given on the CLI.
- **Resumable**: one result file per plan under `eval/cache/ocr/<folder>/<id>.json`. A plan
  with an existing cache entry is skipped unless `--force`. Killing and restarting must lose
  at most the plan in flight.
- **Progress**: tqdm to stderr with rate and ETA; `--workers N` for parallelism.
- Failures are recorded as a cache entry with an `error` field, never a crash, and are counted
  in the report rather than silently dropped.
- Writes `eval/results/tier1_ocr.md` + `.json` reporting, overall and **broken down by folder**:
  - plans processed, failed, cached
  - token count distribution per plan
  - **how many plans have usable text**, against a stated threshold. Define "usable" explicitly
    as: >= 3 tokens above the confidence floor that match a known Finnish room label. Report
    the threshold in the output so the number is interpretable, and also report the raw count
    at a couple of nearby thresholds so the choice is visible rather than tuned.
  - how many have any parseable area or dimension token
  - decimal separator counts (comma vs period) actually observed
  - mean OCR latency per page
- This tier's output is what selects the plans used in Tiers 3 and 4. It runs first.

### Tier 2 — ground truth from `model.svg` (free, automatic, whole dataset)
`eval/tier2_ground_truth.py`

- For every plan, parses `model.svg` into `eval/ground_truth/auto/<folder>/<id>.json`:
  room type (from the `Space <Type>` class), Finnish name label, polygon, bbox,
  and area from the shoelace formula at 100 units/m.
- Gives **room type and room count** ground truth for the entire dataset without hand-labelling.
- Also emits the per-plan median scale ratio and flags plans deviating from 100 u/m.
- Writes `eval/results/tier2_ground_truth.md`: room-type distribution, plans with unparseable
  SVG, flagged-scale plans, `Undefined` share.
- This is **automatic** ground truth. It is labelled as such everywhere it is used, and it is
  never described as hand-verified. Areas here are derived from the annotation polygon, not
  read off the drawing.

### Tier 3 — Claude approaches on a 50-plan sample, cost-gated
`eval/tier3_llm.py`

Runs `rules`, `ocr+llm`, `vlm` and `hybrid` over a **50-plan random sample** of the
`high_quality_architectural` test plans, drawn with a documented fixed seed
(`SAMPLE_SEED` in `eval/tier3_cost_probe.py`) so the sample is reproducible. Reuses the
Tier 1 OCR cache so no OCR is recomputed.

**`rules` (OCR + parser + bbox pairing, no model call) is included in every results table.**
It costs nothing and is deterministic, so it is the floor each paid approach must beat.

The full run uses the **Message Batches API** (50% of standard rates). A hard `--max-spend`
cap stops the run before any call that could take the running total past it, checked from
logged costs *before* the call is made.

**API keys are read from the project `.env` file only, never from the process environment.**

`colorful` and `high_quality` are **excluded from all LLM runs and accuracy metrics**: the
former has no text on the page at all, the latter has room labels but essentially no areas or
dimensions, so neither can support a dimension/area metric and paying for LLM calls on them
would buy nothing. They stay in Tier 1 reporting, and `results.md` states this exclusion and
the reason next to every affected table. LLM responses are cached per (plan, approach, model) under
`eval/cache/llm/` and are resumable on the same terms as Tier 1.

**Cost gate — this is a hard stop:**
1. Run the first **20 test plans only** (`--limit 20`, deterministic order, recorded in the
   output so the sample is reproducible).
2. Write `eval/results/tier3_cost_probe.md` from the `llm_calls` table and
   `eval/pricing.yaml`: measured input/output tokens per page, **measured cost per page for
   each approach separately**, measured latency per page, and the extrapolated cost of the
   full ~270-plan run for both approaches.
3. **Stop. Report the measured numbers and wait for explicit approval before the remaining
   ~250 plans.** Do not continue automatically, and do not treat `--limit 20` finishing
   cleanly as approval. The full run is gated behind an explicit `--approved-budget` flag so
   it cannot start by accident.

Pricing comes from `eval/pricing.yaml` (per-model input/output rates, with the date the rates
were taken). Cost is computed from logged tokens; it is never hardcoded in pipeline logic and
never estimated when a real token count is available.

### Tier 4 — hand-verified gold set for dimensions and areas
`eval/tier4_gold.py` (selection) + `eval/label_helper.py` (labelling CLI)

- Selects **15 test-split plans with readable text**, ranked by the Tier 1 signal:
  prefers `high_quality_architectural`, requires parseable area/dimension tokens, and spreads
  across room counts so the set is not all studios. Writes the chosen IDs and the reason each
  was chosen to `eval/data/gold_15.txt` so the selection is auditable and reproducible.
- `label_helper.py` is the labelling CLI I will actually use:
  - Renders the page with OCR boxes drawn to `eval/labelling/<id>.png` for reference.
  - **Pre-fills each room from Tier 2**: SVG room type, Finnish label, and polygon area.
  - Walks room by room, showing the pre-filled value and the OCR tokens near that room's
    polygon, and asks only for confirm / correct / skip. Typing is the exception, not the rule.
  - Writes `eval/ground_truth/gold/<id>.json` with `"verified": true` per field, so a field
    that was accepted from the SVG is distinguishable from one a human actually read off
    the drawing.
  - Resumable per plan and per room; re-running continues where it stopped.
  - `--review <id>` re-opens a finished plan for correction.

Only this tier's numbers are described as hand-verified in the README.

### Metrics
`eval/run_eval.py` consumes the tier outputs and writes `eval/results.md`.

Per approach (`ocr+llm` vs `vlm`) and per preprocessing config:
- OCR token recall for room labels (vs Tier 2 name labels)
- room detection precision / recall / F1, fuzzy name match (vs Tier 2,
  `high_quality_architectural` test plans)
- room-type accuracy (vs Tier 2, `Undefined` excluded and reported separately)
- dimension and area accuracy within 5% tolerance — **Tier 4 gold set only**, n=15, and the
  n is printed next to every number
- accuracy per dimension format, using the format tag from `parse_dims.py`
- mean latency and measured cost per page from `llm_calls` + `eval/pricing.yaml`

`eval/results.md` includes a failure-case section: where each approach breaks and why, with
the plan IDs, including the `colorful` no-text pages and rotated-label failures.

Every table states its sample size and whether its ground truth is automatic (Tier 2) or
hand-verified (Tier 4). Numbers go in the README exactly as measured. If a tier has not been
run, its numbers are absent — never placeholders, never estimates.

## Build phases — stop after each, summarize, and wait for my go-ahead
- **Phase 0**: Scaffold repo, docker-compose with db/api/web, health endpoint, `.env.example`, Alembic init. Verify `docker compose up` works.
- **Phase 1**: Ingest + preprocess + OCR, runnable from a CLI (`python -m app.pipeline.run path/to/plan.pdf`) printing tokens. Save a debug image with boxes drawn.
- **Phase 2**: Dimension parser with full unit tests. Typed results
  (`area` / `dimension_pair` / `door_window_code` / `unknown`) each carrying a reason, the
  matched format for per-format eval reporting, and a plausibility flag.
- **Phase 3**: Three approaches (`ocr+llm`, `vlm`, `hybrid`) over one Structured Outputs
  schema, with grounding checks, Pydantic validation, call logging and DB persistence.
  Gated by a cost probe on the 6 validation plans before any larger run.
- **Phase 4**: API routes + background processing + export.
- **Phase 5**: Next.js upload + viewer.
- **Phase 6**: Eval harness, built and run tier by tier.
  - 6a: Tier 1 OCR sweep (resumable, cached, progress). Report usable-text counts per folder.
  - 6b: Tier 2 ground truth from `model.svg`. Report room-type distribution.
  - 6c: Tier 3 cost probe on 20 test plans. **Report measured cost per page and stop for approval**
        before the remaining 379.
  - 6d: Tier 4 gold-set selection + labelling helper, handed over for me to label.
  - 6e: `run_eval.py` + `results.md` once the gold set exists.
- **Phase 7**: README (Mermaid architecture diagram, setup, eval results, known limitations, how to scale: S3, job queue, batching, GPU OCR), demo GIF instructions.
- **Phase 8** (do not start until I say so): local detector as a third approach.
  - Train a small PyTorch detector on the CubiCasa5K COCO annotations.
  - Export to ONNX, run it in the pipeline with ONNX Runtime (CPU), `source="detector"`.
  - Compare against `ocr+llm` and `vlm` in `eval/results.md` on accuracy, latency and cost
    (the detector's marginal cost per page is zero, which is the point of the comparison).
  - **Classes: `wall`, `room`, `door`, `window`** — the union of the supplied COCO
    annotations and geometry derived from `model.svg`.
  - **Step 8a, SVG -> COCO conversion, is its own deliverable** (`eval/svg_to_coco.py`),
    done and verified before any training:
    - The supplied COCO files carry only `wall` and `room` (verified across all three
      splits: train 173023 anns, val 15926, test 16818, categories `['wall','room']`).
      Door and window boxes do not exist there and must be generated.
    - `model.svg` carries `Door Swing *` and `Window *` geometry. Convert those to boxes and
      merge with the existing `wall`/`room` annotations into a 4-class COCO file.
    - The SVG is in model space at 100 units/m; COCO boxes are in `F1_original.png` pixel
      space. The conversion must derive and verify the per-plan transform, not assume one,
      and drop plans where it cannot be verified rather than emit silently wrong boxes.
    - Validate by rendering boxes over the page for a sample and eyeballing them, and report
      how many plans converted, how many were dropped, and why.
    - Write the converted set to a new file. **Never rewrite the supplied annotations.**
  - Also note: COCO `file_name` values are absolute Kaggle paths
    (`/kaggle/input/cubicasa5k/...`). Remap to `DATASET_DIR` at load time; do not rewrite the
    annotation files.

## Rules
- Plan before coding each phase; list files you'll touch.
- Small, focused commits with clear messages per phase.
- Type hints everywhere; ruff-clean Python; strict TypeScript.
- Handle errors explicitly; failed jobs set status=failed with a message.
- No secrets in code. No hardcoded paths or model names.
- If a dependency fails to install (PaddleOCR is heavy), stop and tell me, with options, instead of silently switching.
- Don't invent accuracy numbers, sample outputs, or benchmark claims anywhere.
- Explain non-obvious CV decisions in code comments — I need to defend them in interviews.
