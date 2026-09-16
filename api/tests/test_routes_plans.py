"""API route tests.

The database is a real Postgres in CI/dev; here the routes are tested against a session
fixture backed by SQLite so the suite stays fast and needs no container. The one thing that
does not translate is the Postgres UUID column type, so the models are mapped to a
SQLite-compatible variant for the test engine only.
"""

import io
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import CHAR, TypeDecorator

from app.db import Base, get_db
from app.main import app
from app.models import Page, Plan, Room


class SqliteUUID(TypeDecorator):
    """Store a UUID as a 36-char string. SQLite has no native UUID type."""

    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else str(value)

    def process_result_value(self, value, dialect):
        return None if value is None else uuid.UUID(str(value))


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    # Swap the Postgres UUID type for the SQLite-safe one on the test metadata only.
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, PgUUID):
                column.type = SqliteUUID()

    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = TestingSession()

    def override():
        yield session

    app.dependency_overrides[get_db] = override
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    yield session
    app.dependency_overrides.clear()
    session.close()


@pytest.fixture()
def client(db_session):
    return TestClient(app)


def make_plan(session, status="done", filename="plan.png"):
    plan = Plan(filename=filename, status=status)
    session.add(plan)
    session.flush()
    page = Page(plan_id=plan.id, page_number=1, image_path=f"plans/{plan.id}/page_001.png",
                width=800, height=600)
    session.add(page)
    session.flush()
    session.add(Room(
        page_id=page.id, name="MH", room_type="bedroom", area_m2=11.7,
        width_m=None, length_m=None, source="rules", raw_text="MH",
        x1=10, y1=20, x2=40, y2=50, confidence=0.9, source_token_ids="0,1", grounded=True,
    ))
    session.add(Room(
        page_id=page.id, name="OH", room_type="living_room", area_m2=22.6,
        width_m=None, length_m=None, source="hybrid", raw_text="OH",
        x1=60, y1=20, x2=90, y2=50, confidence=0.8, source_token_ids=None, grounded=False,
    ))
    session.commit()
    return plan, page


def png_bytes(width=60, height=40):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# --- upload validation -----------------------------------------------------------------


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/plans", files={"file": ("plan.tiff", b"data", "image/tiff")}
    )
    assert response.status_code == 415
    assert "unsupported file type" in response.json()["detail"]


def test_upload_rejects_unsupported_content_type(client):
    response = client.post(
        "/plans", files={"file": ("plan.png", b"data", "application/zip")}
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post("/plans", files={"file": ("plan.png", b"", "image/png")})
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    oversized = b"x" * (2 * 1024 * 1024)
    response = client.post("/plans", files={"file": ("plan.png", oversized, "image/png")})
    get_settings.cache_clear()
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_upload_accepts_png_and_returns_202_with_id(client, monkeypatch):
    """The upload must return immediately; processing happens in the background."""
    called = {}

    def fake_process(plan_id, source_path):
        called["plan_id"] = plan_id

    monkeypatch.setattr("app.routes.plans.process_plan", fake_process)
    response = client.post(
        "/plans", files={"file": ("plan.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 202
    body = response.json()
    assert uuid.UUID(body["id"])
    assert body["status"] == "pending"
    assert called["plan_id"] == uuid.UUID(body["id"])


# --- status ----------------------------------------------------------------------------


def test_get_plan_returns_pages_and_rooms(client, db_session):
    plan, _ = make_plan(db_session)
    body = client.get(f"/plans/{plan.id}").json()
    assert body["status"] == "done"
    assert len(body["pages"]) == 1
    assert {r["name"] for r in body["pages"][0]["rooms"]} == {"MH", "OH"}


def test_get_plan_filters_by_source(client, db_session):
    plan, _ = make_plan(db_session)
    body = client.get(f"/plans/{plan.id}", params={"source": "rules"}).json()
    rooms = body["pages"][0]["rooms"]
    assert [r["name"] for r in rooms] == ["MH"]


def test_get_plan_exposes_grounded_flag(client, db_session):
    """An ungrounded room stays visible and flagged rather than disappearing."""
    plan, _ = make_plan(db_session)
    body = client.get(f"/plans/{plan.id}").json()
    grounded = {r["name"]: r["grounded"] for r in body["pages"][0]["rooms"]}
    assert grounded == {"MH": True, "OH": False}


def test_get_plan_404_for_unknown_id(client):
    assert client.get(f"/plans/{uuid.uuid4()}").status_code == 404


def test_failed_plan_reports_its_error(client, db_session):
    plan = Plan(filename="bad.pdf", status="failed", error="PDF has no pages")
    db_session.add(plan)
    db_session.commit()
    body = client.get(f"/plans/{plan.id}").json()
    assert body["status"] == "failed"
    assert body["error"] == "PDF has no pages"


# --- page image ------------------------------------------------------------------------


def test_page_image_served_from_storage(client, db_session, tmp_path, monkeypatch):
    from app.config import get_settings
    from app.storage import LocalStorage

    get_settings.cache_clear()
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()

    plan, page = make_plan(db_session)
    LocalStorage(tmp_path / "storage").put(page.image_path, png_bytes())

    response = client.get(f"/plans/{plan.id}/pages/1/image")
    get_settings.cache_clear()
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_image_404_for_missing_page(client, db_session):
    plan, _ = make_plan(db_session)
    assert client.get(f"/plans/{plan.id}/pages/99/image").status_code == 404


# --- export ----------------------------------------------------------------------------


def test_export_csv_has_a_header_and_a_row_per_room(client, db_session):
    plan, _ = make_plan(db_session)
    response = client.get(f"/plans/{plan.id}/export", params={"format": "csv"})
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

    import csv as csv_module

    rows = list(csv_module.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2
    assert rows[0]["room_type"] == "bedroom"
    assert rows[0]["area_m2"] == "11.7"
    # area_m2 is primary; width/length stay empty rather than being derived.
    assert rows[0]["width_m"] == ""
    assert rows[0]["length_m"] == ""


def test_export_csv_filename_is_attached(client, db_session):
    plan, _ = make_plan(db_session, filename="villa.pdf")
    response = client.get(f"/plans/{plan.id}/export", params={"format": "csv"})
    assert "villa_takeoff.csv" in response.headers["content-disposition"]


def test_export_json(client, db_session):
    plan, _ = make_plan(db_session)
    body = client.get(f"/plans/{plan.id}/export", params={"format": "json"}).json()
    assert body["plan_id"] == str(plan.id)
    assert len(body["rooms"]) == 2
    assert body["rooms"][0]["page_number"] == 1


def test_export_filters_by_source(client, db_session):
    plan, _ = make_plan(db_session)
    body = client.get(
        f"/plans/{plan.id}/export", params={"format": "json", "source": "hybrid"}
    ).json()
    assert [r["name"] for r in body["rooms"]] == ["OH"]
    assert body["source_filter"] == "hybrid"


def test_export_rejects_unknown_format(client, db_session):
    plan, _ = make_plan(db_session)
    assert client.get(
        f"/plans/{plan.id}/export", params={"format": "xlsx"}
    ).status_code == 422


def test_export_of_a_plan_with_no_rooms_is_an_empty_table_not_an_error(client, db_session):
    plan = Plan(filename="empty.png", status="done")
    db_session.add(plan)
    db_session.commit()
    response = client.get(f"/plans/{plan.id}/export", params={"format": "csv"})
    assert response.status_code == 200
    assert response.text.strip().splitlines()[0].startswith("plan_id,")


# --- tokens ----------------------------------------------------------------------------


def test_tokens_endpoint_returns_indices_the_model_cites(client, db_session):
    from app.models import OcrTokenRow

    plan, page = make_plan(db_session)
    db_session.add(OcrTokenRow(
        page_id=page.id, token_index=0, text="MH", confidence=0.9,
        x1=10, y1=20, x2=40, y2=50, angle=90,
    ))
    db_session.commit()
    body = client.get(f"/plans/{plan.id}/tokens").json()
    assert body == [{"id": 0, "text": "MH", "confidence": 0.9, "bbox": [10, 20, 40, 50]}]


# --- CORS ------------------------------------------------------------------------------


def test_cors_allows_the_web_origin(client):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
