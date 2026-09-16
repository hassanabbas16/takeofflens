# TakeoffLens — Build Spec

## What we're building
A blueprint analysis tool. User uploads an architectural floor plan (PDF or image). The system extracts text (room labels, dimensions, annotations) with OCR, parses it into structured takeoff data (rooms, dimensions, areas, counts), and shows the results in a web viewer with bounding-box overlays and CSV/JSON export.

This is a portfolio project for an AI/CV full-stack role. It must actually work end-to-end, be runnable with one command, and include an honest evaluation with real numbers. Never fabricate metrics or fake results.

## Stack (do not substitute without asking)
- Backend: Python 3.11, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2
- CV/OCR: OpenCV, PaddleOCR (EasyOCR as a documented fallback if Paddle install fails), PyMuPDF for PDF -> image
- LLM/VLM: OpenAI API via the official `openai` Python SDK
  - Key from env `OPENAI_API_KEY`
  - Model names from env: `OPENAI_TEXT_MODEL` and `OPENAI_VISION_MODEL` (never hardcode; check the current OpenAI docs for suitable vision-capable models and put sensible defaults in `.env.example`)
  - Use Structured Outputs (JSON schema response format, generated from the Pydantic models) for all extraction calls
- DB: PostgreSQL 16
- Frontend: Next.js (App Router), TypeScript, Tailwind CSS
- Infra: Docker + docker-compose (api, web, db)
- Tests: pytest (backend)

## Repo structure
```
takeofflens/
  api/
    app/
      main.py
      config.py            # pydantic-settings, env vars
      db.py
      models.py            # SQLAlchemy models
      schemas.py           # Pydantic schemas (also used as OpenAI response schemas)
      llm.py               # thin OpenAI client wrapper: retries, timeouts, token/cost logging
      routes/
        plans.py
      pipeline/
        ingest.py          # PDF/image -> normalized page images
        preprocess.py      # OpenCV steps
        ocr.py             # OCR wrapper, returns text + bbox + confidence
        parse_dims.py      # regex dimension parser -> meters + area
        classify.py        # OpenAI text model: OCR tokens -> structured rooms JSON
        vlm_direct.py      # OpenAI vision model baseline (image -> rooms JSON)
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
    data/                  # sample plans (gitignored if large) + README on source
    ground_truth/          # hand-labeled JSON per plan
    run_eval.py            # computes metrics, writes eval/results.md
  docker-compose.yml
  .env.example
  README.md
```

## Data model
- `plans`: id (uuid), filename, status (pending|processing|done|failed), error, created_at
- `pages`: id, plan_id, page_number, image_path, width, height
- `ocr_tokens`: id, page_id, text, confidence, bbox (x1,y1,x2,y2 as ints)
- `rooms`: id, page_id, name, room_type, width_m, length_m, area_m2, source ("ocr+llm" | "vlm"), raw_text, bbox (nullable)
- `llm_calls`: id, page_id, purpose, model, input_tokens, output_tokens, latency_ms, created_at (for cost/latency reporting)

## Pipeline requirements
1. **Ingest**: PDF pages rendered at 300 DPI via PyMuPDF; images passed through. Store page images to a local `storage/` volume behind a `Storage` interface so S3 can be swapped in later.
2. **Preprocess** (each step toggleable via config, for eval ablations): grayscale, denoise, adaptive threshold, deskew, upscale if the short side < 2000px.
3. **OCR**: return a list of `{text, confidence, bbox}`. Handle rotated text (PaddleOCR angle classifier on). Drop tokens below a configurable confidence.
4. **Dimension parsing**: regex + unit normalization. Must handle at minimum: `12'6" x 10'`, `12' - 6" x 10' - 0"`, `3.5 x 4.2`, `3.5m x 4.2m`, `3500 x 4200` (mm). Convert all to meters, compute area. Unit tests for every format plus malformed input.
5. **Classify**: send OCR tokens (text + bbox) to the OpenAI text model with a strict JSON schema. Associate each room label with its nearest dimension string using bbox proximity *before* the LLM call, and pass candidate pairs in the prompt. Validate with Pydantic; retry once on failure; never crash the job. Temperature 0.
6. **VLM baseline**: send the page image (base64, downscaled to a sensible max size to control cost) directly to the OpenAI vision model, same output schema. Store with `source="vlm"` so the two approaches can be compared.
7. Log every OpenAI call to `llm_calls` (tokens, latency). Cost per page is computed in eval from a pricing config file, not hardcoded in logic.
8. Processing runs as a FastAPI BackgroundTask for now. Keep the interface clean enough to move to a queue later; note that in the README.

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

## Evaluation (this is what makes the project credible)
- `eval/data/`: 10–20 floor plans. Use CubiCasa5K or public-domain plans; document source and license in `eval/README.md`.
- `eval/ground_truth/`: I will hand-label rooms (name + dimensions) per plan. Create a labeling template and a small CLI helper to speed it up.
- `run_eval.py` reports, per approach (ocr+llm vs vlm) and per preprocessing config:
  - OCR token recall for room labels
  - room detection precision/recall/F1 (fuzzy name match)
  - dimension accuracy (within 5% tolerance)
  - mean latency and approx cost per page (from `llm_calls` + pricing config)
- Writes `eval/results.md` with a table plus a failure-case section (where each approach breaks and why).
- Numbers go in the README exactly as measured.

## Build phases — stop after each, summarize, and wait for my go-ahead
- **Phase 0**: Scaffold repo, docker-compose with db/api/web, health endpoint, `.env.example`, Alembic init. Verify `docker compose up` works.
- **Phase 1**: Ingest + preprocess + OCR, runnable from a CLI (`python -m app.pipeline.run path/to/plan.pdf`) printing tokens. Save a debug image with boxes drawn.
- **Phase 2**: Dimension parser with full unit tests.
- **Phase 3**: OpenAI classification + VLM baseline, Structured Outputs, Pydantic validation, call logging, DB persistence.
- **Phase 4**: API routes + background processing + export.
- **Phase 5**: Next.js upload + viewer.
- **Phase 6**: Eval harness + results.md.
- **Phase 7**: README (Mermaid architecture diagram, setup, eval results, known limitations, how to scale: S3, job queue, batching, GPU OCR), demo GIF instructions.

## Rules
- Plan before coding each phase; list files you'll touch.
- Small, focused commits with clear messages per phase.
- Type hints everywhere; ruff-clean Python; strict TypeScript.
- Handle errors explicitly; failed jobs set status=failed with a message.
- No secrets in code. No hardcoded paths or model names.
- If a dependency fails to install (PaddleOCR is heavy), stop and tell me, with options, instead of silently switching.
- Don't invent accuracy numbers, sample outputs, or benchmark claims anywhere.
- Explain non-obvious CV decisions in code comments — I need to defend them in interviews.
