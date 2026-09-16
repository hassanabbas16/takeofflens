import cv2
import numpy as np
import pytest

from app.pipeline.ingest import IngestError, ingest
from app.pipeline.preprocess import PreprocessConfig, estimate_skew_angle, preprocess
from app.storage import LocalStorage


def _write_png(path, width=120, height=80):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MH", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)
    return img


def test_ingest_png(tmp_path):
    path = tmp_path / "plan.png"
    _write_png(path)
    pages = ingest(path)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert (pages[0].width, pages[0].height) == (120, 80)
    assert pages[0].image.ndim == 3


def test_ingest_missing_file(tmp_path):
    with pytest.raises(IngestError, match="file not found"):
        ingest(tmp_path / "nope.png")


def test_ingest_unsupported_suffix(tmp_path):
    path = tmp_path / "plan.tiff"
    path.write_bytes(b"not really a tiff")
    with pytest.raises(IngestError, match="unsupported file type"):
        ingest(path)


def test_ingest_corrupt_image(tmp_path):
    path = tmp_path / "plan.png"
    path.write_bytes(b"definitely not a png")
    with pytest.raises(IngestError, match="could not decode"):
        ingest(path)


def test_ingest_pdf_renders_at_dpi(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "plan.pdf"
    doc = fitz.open()
    # 72pt x 72pt page = exactly 1 inch square, so page size in px == dpi.
    page = doc.new_page(width=72, height=72)
    page.insert_text((10, 40), "MH 11.7")
    doc.save(path)
    doc.close()

    pages = ingest(path, dpi=150)
    assert len(pages) == 1
    assert pages[0].width == pytest.approx(150, abs=2)
    assert pages[0].image.ndim == 3


def test_preprocess_always_returns_three_channels():
    """PaddleOCR crashes on single-channel input, so every config must return BGR."""
    img = np.full((60, 80, 3), 255, dtype=np.uint8)
    for config in (
        PreprocessConfig(),
        PreprocessConfig(grayscale=True),
        PreprocessConfig(denoise=True),
        PreprocessConfig(adaptive_threshold=True),
        PreprocessConfig(grayscale=True, denoise=True, adaptive_threshold=True),
    ):
        out = preprocess(img, config)
        assert out.ndim == 3, f"{config.applied} returned {out.ndim} channels"
        assert out.shape[2] == 3


def test_preprocess_noop_by_default_preserves_image():
    img = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    assert np.array_equal(preprocess(img, PreprocessConfig()), img)


def test_preprocess_upscale_respects_min_short_side():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = preprocess(img, PreprocessConfig(upscale=True, upscale_min_short_side=400))
    assert min(out.shape[:2]) >= 400


def test_preprocess_upscale_skips_large_images():
    img = np.full((500, 600, 3), 255, dtype=np.uint8)
    out = preprocess(img, PreprocessConfig(upscale=True, upscale_min_short_side=400))
    assert out.shape[:2] == (500, 600)


def test_preprocess_config_applied_lists_enabled_steps():
    config = PreprocessConfig(grayscale=True, deskew=True)
    assert config.applied == ["grayscale", "deskew"]
    assert PreprocessConfig().applied == []


def test_estimate_skew_on_straight_lines_is_near_zero():
    img = np.full((200, 400), 255, dtype=np.uint8)
    for y in (50, 100, 150):
        cv2.line(img, (10, y), (390, y), 0, 2)
    assert abs(estimate_skew_angle(img)) < 1.0


def test_estimate_skew_detects_tilt():
    img = np.full((200, 400), 255, dtype=np.uint8)
    for y in (50, 100, 150):
        cv2.line(img, (10, y), (390, y), 0, 2)
    matrix = cv2.getRotationMatrix2D((200, 100), 5.0, 1.0)
    tilted = cv2.warpAffine(img, matrix, (400, 200), borderValue=255)
    # cv2 rotates counter-clockwise for positive angles, so the measured skew is negative.
    assert estimate_skew_angle(tilted) == pytest.approx(-5.0, abs=1.5)


def test_estimate_skew_blank_page_returns_zero():
    assert estimate_skew_angle(np.full((100, 100), 255, dtype=np.uint8)) == 0.0


def test_local_storage_round_trip(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    storage.put("plans/a/page1.png", b"bytes")
    assert storage.exists("plans/a/page1.png")
    assert storage.get("plans/a/page1.png") == b"bytes"
    assert storage.local_path("plans/a/page1.png").is_file()


def test_local_storage_rejects_traversal(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.put("../escaped.txt", b"nope")


def test_local_storage_put_file(tmp_path):
    source = tmp_path / "src.png"
    source.write_bytes(b"image-bytes")
    storage = LocalStorage(tmp_path / "storage")
    storage.put_file("plans/b/page1.png", source)
    assert storage.get("plans/b/page1.png") == b"image-bytes"
