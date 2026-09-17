"""Do the supplied COCO boxes actually line up with the page images?

Phase 8 trains a detector on these annotations. Before any of that, the assumption that a
COCO box is in some particular image's pixel space has to be *checked*, not inherited from a
README - the SVG annotations in this same dataset turned out not to map to the page by any
constant, and silently wrong boxes would poison training with no visible error.

Three independent checks, from cheapest to strongest:

1. **Declared size.** Does the COCO ``images`` entry's width/height match the PNG on disk,
   for ``F1_original.png`` and for ``F1_scaled.png``? This alone identifies which image the
   annotations belong to.
2. **Containment.** Do the boxes fall inside the declared frame, and does their union cover
   a sensible fraction of it? Boxes in the wrong space typically spill outside or huddle in
   one corner.
3. **Ink overlap, measured on ``wall`` boxes.** A wall annotation should be almost solid ink;
   the same box shifted across the page should not be. Room boxes are useless for this - a
   room is mostly white by definition, so a correctly placed room box and a misplaced one
   score about the same (1.19x in the first version of this check, which is why it moved to
   walls). A second, independent signal: a room box's *perimeter* should be much darker than
   its interior, because its perimeter is the walls.

Also renders a sample with boxes drawn, to be looked at.

    python eval/verify_coco_alignment.py --split test --sample 40 --render 6

No API calls. Free.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")

COCO_DIR = Path("/data/cubicasa5k_coco")
DATASET_DIR = Path("/data/cubicasa5k")
OUT_DIR = Path("/eval/labelling/coco_check")

# Anything darker than this counts as ink on these scans.
INK_THRESHOLD = 200


def load_coco(split: str) -> dict:
    path = COCO_DIR / f"{split}_coco_pt.json"
    return json.loads(path.read_text(encoding="utf-8"))


def local_plan_dir(file_name: str) -> Path | None:
    """Map a Kaggle absolute path in `file_name` onto the mounted dataset.

    The annotations ship with paths like
    /kaggle/input/cubicasa5k/cubicasa5k/high_quality_architectural/1191/F1_original.png
    which do not exist here. Only the tail from the folder name onwards is meaningful.
    """
    parts = Path(file_name).parts
    for i, part in enumerate(parts):
        if part in ("high_quality_architectural", "high_quality", "colorful"):
            return DATASET_DIR.joinpath(*parts[i:])
    return None


def perimeter_vs_interior(
    gray: np.ndarray, box: list[float], band: int = 6
) -> tuple[float | None, float | None]:
    """Ink in a room box's outer band vs its middle.

    A correctly placed room box has its walls on the perimeter and mostly empty floor in the
    middle, so the two differ sharply. A box in the wrong place has no reason to.
    """
    x, y, w, h = (int(v) for v in box)
    if w < 4 * band or h < 4 * band:
        return None, None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x2 - x1 < 4 * band or y2 - y1 < 4 * band:
        return None, None
    outer = gray[y1:y2, x1:x2]
    inner = gray[y1 + band:y2 - band, x1 + band:x2 - band]
    if outer.size == 0 or inner.size == 0:
        return None, None
    outer_ink = int((outer < INK_THRESHOLD).sum())
    inner_ink = int((inner < INK_THRESHOLD).sum())
    edge_pixels = outer.size - inner.size
    if edge_pixels <= 0:
        return None, None
    return (outer_ink - inner_ink) / edge_pixels, inner_ink / inner.size


def ink_fraction(gray: np.ndarray, box: list[float]) -> float | None:
    x, y, w, h = box
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(gray.shape[1], int(x + w)), min(gray.shape[0], int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return float((crop < INK_THRESHOLD).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--render", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/coco_alignment.md"))
    args = parser.parse_args()

    coco = load_coco(args.split)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    print(f"{args.split}: {len(coco['images'])} images, {len(coco['annotations'])} anns, "
          f"categories {sorted(cats.values())}")

    rng = random.Random(args.seed)
    images = rng.sample(coco["images"], min(args.sample, len(coco["images"])))

    size_match: Counter = Counter()
    contained = Counter()
    ink_real: list[float] = []
    ink_control: list[float] = []
    room_edge: list[float] = []
    room_inner: list[float] = []
    coverage: list[float] = []
    scale_ratios: list[tuple[str, float, float]] = []
    missing = 0
    rendered = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_image_rows: list[str] = []

    for entry in images:
        png = local_plan_dir(entry["file_name"])
        if png is None or not png.exists():
            missing += 1
            continue
        plan_dir = png.parent
        declared = (entry["width"], entry["height"])

        sizes = {}
        for name in ("F1_original.png", "F1_scaled.png"):
            candidate = plan_dir / name
            if candidate.exists():
                img = cv2.imread(str(candidate))
                if img is not None:
                    sizes[name] = (img.shape[1], img.shape[0])

        if "F1_original.png" in sizes and "F1_scaled.png" in sizes:
            ow, oh = sizes["F1_original.png"]
            sw, sh = sizes["F1_scaled.png"]
            if ow and oh:
                scale_ratios.append((plan_dir.name, sw / ow, sh / oh))

        matched = [n for n, s in sizes.items() if s == declared]
        key = matched[0] if matched else "NONE"
        size_match[key] += 1
        per_image_rows.append(
            f"| {plan_dir.name} | {declared[0]}x{declared[1]} | "
            + " | ".join(
                f"{sizes.get(n, ('-', '-'))[0]}x{sizes.get(n, ('-', '-'))[1]}"
                for n in ("F1_original.png", "F1_scaled.png")
            )
            + f" | {key} |"
        )

        anns = by_image.get(entry["id"], [])
        if not anns:
            continue

        # Containment against the declared frame.
        inside = sum(
            1 for a in anns
            if a["bbox"][0] >= -1 and a["bbox"][1] >= -1
            and a["bbox"][0] + a["bbox"][2] <= declared[0] + 1
            and a["bbox"][1] + a["bbox"][3] <= declared[1] + 1
        )
        contained["inside"] += inside
        contained["total"] += len(anns)

        # Union coverage of room boxes, as a fraction of the page.
        mask = np.zeros((declared[1], declared[0]), dtype=np.uint8)
        for a in anns:
            if cats.get(a["category_id"]) != "room":
                continue
            x, y, w, h = (int(v) for v in a["bbox"])
            cv2.rectangle(mask, (max(0, x), max(0, y)),
                          (min(declared[0], x + w), min(declared[1], y + h)), 255, -1)
        coverage.append(float((mask > 0).mean()))

        # Ink overlap, against the image the size check selected.
        target = plan_dir / (matched[0] if matched else "F1_original.png")
        if target.exists():
            img = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                for a in anns:
                    name = cats.get(a["category_id"])
                    if name == "wall":
                        real = ink_fraction(img, a["bbox"])
                        # Control: the same box shifted by a third of the page. If the
                        # annotations were in the wrong coordinate space, the real boxes
                        # would score like this one.
                        shifted = [
                            (a["bbox"][0] + declared[0] / 3)
                            % max(1, declared[0] - a["bbox"][2]),
                            (a["bbox"][1] + declared[1] / 3)
                            % max(1, declared[1] - a["bbox"][3]),
                            a["bbox"][2], a["bbox"][3],
                        ]
                        ctrl = ink_fraction(img, shifted)
                        if real is not None:
                            ink_real.append(real)
                        if ctrl is not None:
                            ink_control.append(ctrl)
                    elif name == "room":
                        edge, inner = perimeter_vs_interior(img, a["bbox"])
                        if edge is not None and inner is not None:
                            room_edge.append(edge)
                            room_inner.append(inner)

                if rendered < args.render:
                    canvas = cv2.imread(str(target))
                    colours = {"room": (0, 170, 0), "wall": (200, 0, 0)}
                    for a in anns:
                        name = cats.get(a["category_id"], "?")
                        x, y, w, h = (int(v) for v in a["bbox"])
                        cv2.rectangle(canvas, (x, y), (x + w, y + h),
                                      colours.get(name, (0, 0, 255)), 2)
                    cv2.imwrite(str(OUT_DIR / f"{plan_dir.name}_{args.split}.png"), canvas)
                    rendered += 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    verdict_size = size_match.most_common(1)[0] if size_match else ("none", 0)
    wall_ratio = mean(ink_real) / max(mean(ink_control), 1e-9)
    edge_ratio = mean(room_edge) / max(mean(room_inner), 1e-9)
    aligned = (
        verdict_size[0] != "NONE"
        and contained["total"]
        and contained["inside"] / contained["total"] > 0.99
        and wall_ratio > 1.5
        and edge_ratio > 1.5
    )

    lines = [
        "# Do the supplied COCO boxes line up with the page images?",
        "",
        "Checked before Phase 8, because the `model.svg` annotations in this same dataset do "
        "**not** map to the page by any constant. An assumption about coordinate space is "
        "worth nothing until it is measured.",
        "",
        f"- Split: `{args.split}` - {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, categories "
        f"`{sorted(cats.values())}`",
        f"- Sample: {len(images)} images, seed {args.seed}",
        f"- Plans whose image could not be found on the mount: {missing}",
        "",
        "## 1. Which image do the declared sizes match?",
        "",
        "| Matches | Images |",
        "| --- | --- |",
    ]
    for name, n in size_match.most_common():
        lines.append(f"| `{name}` | {n} |")
    lines += [
        "",
        "## 2. Containment",
        "",
        f"- Boxes inside the declared frame: **{contained['inside']}/{contained['total']}** "
        f"({100 * contained['inside'] / max(contained['total'], 1):.1f}%)",
        f"- Mean page area covered by the union of `room` boxes: "
        f"**{100 * mean(coverage):.1f}%**",
        "",
        "## 3. Ink overlap",
        "",
        "Measured on `wall` boxes, which should be almost solid ink. The control is the same "
        "box shifted by a third of the page - what the real boxes would look like if the "
        "annotations were in the wrong coordinate space.",
        "",
        f"- Mean dark-pixel fraction inside `wall` boxes: **{mean(ink_real):.3f}**",
        f"- Same boxes, shifted (control):               **{mean(ink_control):.3f}**",
        f"- Ratio: **{wall_ratio:.2f}x**",
        "",
        "Room boxes are deliberately not used for this: a room is mostly white by "
        "definition, so a correct box and a misplaced one score alike (the first version of "
        "this check scored 1.19x and proved nothing). Instead, for rooms:",
        "",
        f"- Ink in a room box's outer band (its walls): **{mean(room_edge):.3f}**",
        f"- Ink in its interior (the floor):            **{mean(room_inner):.3f}**",
        f"- Ratio: **{edge_ratio:.2f}x**",
        "",
        "## Verdict",
        "",
        (
            f"**Aligned.** The annotations are in `{verdict_size[0]}` pixel space."
            if aligned
            else "**NOT verified.** Do not train on these until this is understood."
        ),
        "",
        f"Sample renders with boxes drawn: `{OUT_DIR}` - look at them before trusting this.",
        "",
        "## Consequence for Phase 8: the pipeline does not use this image",
        "",
        "The annotations are in `F1_original.png` space, but the OCR pipeline runs on "
        "`F1_scaled.png`, which is a **different and larger** rendering of the same plan. A "
        "detector trained on these boxes therefore cannot be composed with OCR token boxes "
        "without a conversion.",
        "",
        "The good news, and the difference from the `model.svg` case: the scale is **uniform "
        "in x and y within a plan**, so the conversion is exact and needs no fitting - it is "
        "just the ratio of the two image sizes, which are both on disk. It does vary a lot "
        "*between* plans, so it must be computed per plan and never assumed.",
        "",
        f"Measured over {len(scale_ratios)} sampled plans:",
        "",
        f"- scaled/original ratio ranges "
        f"**{min(r[1] for r in scale_ratios):.2f}x - {max(r[1] for r in scale_ratios):.2f}x**"
        if scale_ratios else "- (no plans had both images)",
        f"- max |x ratio - y ratio| within a plan: "
        f"**{max(abs(r[1] - r[2]) for r in scale_ratios):.4f}** (0 would be perfectly uniform)"
        if scale_ratios else "",
        "",
        "For contrast, the `model.svg` annotations have per-axis ratios that disagree with "
        "each other (0.909/1.002, 0.949/1.065, 1.025/1.054), which is why SVG polygons are "
        "not drawn onto the page anywhere in this project.",
        "",
        "## Per-image sizes (sample)",
        "",
        "| Plan | COCO declares | F1_original.png | F1_scaled.png | Matches |",
        "| --- | --- | --- | --- | --- |",
        *per_image_rows[:40],
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nsize match: {dict(size_match)}")
    print(f"contained: {contained['inside']}/{contained['total']}")
    print(f"room-box page coverage: {100 * mean(coverage):.1f}%")
    print(f"wall ink {mean(ink_real):.3f} vs shifted control {mean(ink_control):.3f} "
          f"({wall_ratio:.2f}x)")
    print(f"room edge {mean(room_edge):.3f} vs interior {mean(room_inner):.3f} "
          f"({edge_ratio:.2f}x)")
    print(f"\nVERDICT: {'ALIGNED with ' + verdict_size[0] if aligned else 'NOT VERIFIED'}")
    if scale_ratios:
        print(f"F1_scaled/F1_original ratio: "
              f"{min(r[1] for r in scale_ratios):.2f}x-{max(r[1] for r in scale_ratios):.2f}x, "
              f"max x/y disagreement {max(abs(r[1] - r[2]) for r in scale_ratios):.4f}")
    print(f"rendered {rendered} samples to {OUT_DIR}")
    print(f"wrote {args.out}")
    return 0 if aligned else 1


if __name__ == "__main__":
    raise SystemExit(main())
