"""Pipeline CLI: ingest -> preprocess -> OCR, printing tokens and saving a debug image.

    python -m app.pipeline.run path/to/plan.pdf
    python -m app.pipeline.run /data/cubicasa5k/high_quality_architectural/1191/F1_scaled.png \
        --debug-image out/1191_debug.png --json out/1191.json

DB persistence arrives in Phase 3; this stage is deliberately runnable with no database.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from app.config import get_settings
from app.pipeline.ingest import IngestError, ingest
from app.pipeline.ocr import OcrToken, draw_tokens, run_ocr
from app.pipeline.preprocess import PreprocessConfig, preprocess

logger = logging.getLogger("takeofflens.pipeline")


def _build_preprocess_config(args: argparse.Namespace) -> PreprocessConfig:
    settings = get_settings()
    return PreprocessConfig(
        grayscale=args.grayscale or settings.preprocess_grayscale,
        denoise=args.denoise or settings.preprocess_denoise,
        adaptive_threshold=args.adaptive_threshold or settings.preprocess_adaptive_threshold,
        deskew=args.deskew or settings.preprocess_deskew,
        upscale=args.upscale or settings.preprocess_upscale,
        upscale_min_short_side=settings.preprocess_upscale_min_short_side,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline.run",
        description="Run ingest + preprocess + OCR over a plan and print the tokens.",
    )
    parser.add_argument("path", type=Path, help="PDF, PNG or JPG floor plan")
    parser.add_argument("--dpi", type=int, default=settings.ingest_dpi,
                        help="PDF render DPI (default: %(default)s)")
    parser.add_argument("--lang", default=settings.ocr_lang,
                        help="OCR language (default: %(default)s)")
    parser.add_argument("--orientations", default=settings.ocr_orientations,
                        help="Comma-separated degrees to OCR at (default: %(default)s)")
    parser.add_argument("--min-confidence", type=float, default=settings.ocr_min_confidence,
                        help="Drop tokens below this confidence (default: %(default)s)")
    parser.add_argument("--debug-image", type=Path, default=None,
                        help="Write the page with token boxes drawn to this path")
    parser.add_argument("--json", dest="json_out", type=Path, default=None,
                        help="Write tokens as JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress the token table")

    group = parser.add_argument_group("preprocessing (all default off; see preprocess.py)")
    group.add_argument("--grayscale", action="store_true")
    group.add_argument("--denoise", action="store_true")
    group.add_argument("--adaptive-threshold", action="store_true")
    group.add_argument("--deskew", action="store_true")
    group.add_argument("--upscale", action="store_true")

    return parser.parse_args(argv)


def _print_tokens(tokens: list[OcrToken]) -> None:
    if not tokens:
        print("  (no tokens above the confidence threshold)")
        return
    print(f"  {'text':<32} {'conf':>6}  {'angle':>5}  bbox")
    print(f"  {'-' * 32} {'-' * 6}  {'-' * 5}  {'-' * 24}")
    for token in tokens:
        x1, y1, x2, y2 = token.bbox
        text = token.text if len(token.text) <= 32 else token.text[:29] + "..."
        print(f"  {text:<32} {token.confidence:>6.3f}  {token.angle:>5}  ({x1},{y1},{x2},{y2})")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    settings = get_settings()

    try:
        pages = ingest(args.path, dpi=args.dpi)
    except IngestError as exc:
        # Expected failure mode: report it cleanly rather than dumping a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pre_config = _build_preprocess_config(args)
    orientations = [int(a) for a in args.orientations.split(",") if a.strip()]

    print(f"file: {args.path}")
    print(f"pages: {len(pages)}   preprocess: {pre_config.applied or ['none']}   "
          f"orientations: {orientations}")

    payload: list[dict[str, object]] = []
    for page in pages:
        print(f"\npage {page.page_number}: {page.width}x{page.height}")

        t0 = time.perf_counter()
        processed = preprocess(page.image, pre_config)
        pre_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tokens = run_ocr(
            processed,
            lang=args.lang,
            orientations=orientations,
            det_limit_side_len=settings.ocr_det_limit_side_len,
            det_limit_type=settings.ocr_det_limit_type,
            min_confidence=args.min_confidence,
            dedup_iou=settings.ocr_dedup_iou,
        )
        ocr_ms = (time.perf_counter() - t0) * 1000

        print(f"  preprocess {pre_ms:.0f}ms   ocr {ocr_ms:.0f}ms   tokens {len(tokens)}")
        if not args.quiet:
            _print_tokens(tokens)

        payload.append({
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "preprocess": pre_config.applied,
            "orientations": orientations,
            "ocr_ms": round(ocr_ms),
            "tokens": [t.as_dict() for t in tokens],
        })

        if args.debug_image is not None:
            # Draw on the ORIGINAL page, not the preprocessed one: boxes are in original
            # coordinates, and a thresholded background makes the overlay unreadable.
            out = args.debug_image
            if len(pages) > 1:
                out = out.with_name(f"{out.stem}_p{page.page_number}{out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), draw_tokens(page.image, tokens))
            print(f"  debug image -> {out}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"source": str(args.path), "pages": payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\njson -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
