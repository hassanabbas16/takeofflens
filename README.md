# TakeoffLens

Blueprint analysis: upload an architectural floor plan, extract rooms, dimensions and areas
with OCR + LLM, and review the results with bounding-box overlays.

> **Status: Phase 0 (scaffold).** The pipeline, API routes, viewer and eval land in later
> phases. This README is replaced with the full write-up in Phase 7. No accuracy numbers
> appear here until they have actually been measured.

## Quick start

```bash
cp .env.example .env    # then edit: OPENAI_API_KEY, DATASET_DIR, DATASET_COCO_DIR
docker compose up --build
```

- API: http://localhost:8000/health
- Web: http://localhost:3000

`/health` reports database connectivity and whether the CubiCasa5K dataset mount is present,
so a misconfigured `DATASET_DIR` fails loudly at startup instead of deep inside an eval run.

## Dataset

CubiCasa5K is **not** vendored into this repo. Set `DATASET_DIR` and `DATASET_COCO_DIR` to the
host paths in `.env`; docker-compose mounts both **read-only** into the api container at
`/data/cubicasa5k` and `/data/cubicasa5k_coco`. See [eval/README.md](eval/README.md).

## Local development (without Docker)

```bash
cd api
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check .
```

## Layout

| Path | What |
| --- | --- |
| `api/` | FastAPI service, CV/OCR pipeline, Alembic migrations |
| `web/` | Next.js App Router frontend |
| `eval/` | Tiered evaluation harness and ground truth |

See [CLAUDE.md](CLAUDE.md) for the full build spec.
