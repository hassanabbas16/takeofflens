"""TakeoffLens API entrypoint."""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.db import engine
from app.routes import plans

settings = get_settings()

app = FastAPI(title="TakeoffLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(plans.router)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the two external dependencies we can cheaply verify.

    The dataset check is a read-only existence probe: Phase 6 depends on the mount being
    present, and a missing mount is far cheaper to catch here than 4000 plans into a sweep.
    """
    db_ok = False
    db_error: str | None = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:  # noqa: BLE001 - surfaced in the payload, not swallowed
        db_error = str(exc)

    dataset_dir = settings.dataset_dir
    splits = {
        name: (dataset_dir / f"{name}.txt").exists() for name in ("train", "val", "test")
    }

    return {
        "status": "ok" if db_ok else "degraded",
        "db": {"ok": db_ok, "error": db_error},
        "dataset": {
            "path": str(dataset_dir),
            "mounted": dataset_dir.exists(),
            "splits": splits,
        },
        "coco": {
            "path": str(settings.dataset_coco_dir),
            "mounted": settings.dataset_coco_dir.exists(),
        },
    }
