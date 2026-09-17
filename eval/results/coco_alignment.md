# Do the supplied COCO boxes line up with the page images?

Checked before Phase 8, because the `model.svg` annotations in this same dataset do **not** map to the page by any constant. An assumption about coordinate space is worth nothing until it is measured.

- Split: `test` - 400 images, 16818 annotations, categories `['room', 'wall']`
- Sample: 60 images, seed 20260917
- Plans whose image could not be found on the mount: 0

## 1. Which image do the declared sizes match?

| Matches | Images |
| --- | --- |
| `F1_original.png` | 60 |

## 2. Containment

- Boxes inside the declared frame: **2576/2576** (100.0%)
- Mean page area covered by the union of `room` boxes: **52.1%**

## 3. Ink overlap

Measured on `wall` boxes, which should be almost solid ink. The control is the same box shifted by a third of the page - what the real boxes would look like if the annotations were in the wrong coordinate space.

- Mean dark-pixel fraction inside `wall` boxes: **0.620**
- Same boxes, shifted (control):               **0.119**
- Ratio: **5.23x**

Room boxes are deliberately not used for this: a room is mostly white by definition, so a correct box and a misplaced one score alike (the first version of this check scored 1.19x and proved nothing). Instead, for rooms:

- Ink in a room box's outer band (its walls): **0.238**
- Ink in its interior (the floor):            **0.126**
- Ratio: **1.89x**

## Verdict

**Aligned.** The annotations are in `F1_original.png` pixel space.

Sample renders with boxes drawn: `/eval/labelling/coco_check` - look at them before trusting this.

## Consequence for Phase 8: the pipeline does not use this image

The annotations are in `F1_original.png` space, but the OCR pipeline runs on `F1_scaled.png`, which is a **different and larger** rendering of the same plan. A detector trained on these boxes therefore cannot be composed with OCR token boxes without a conversion.

The good news, and the difference from the `model.svg` case: the scale is **uniform in x and y within a plan**, so the conversion is exact and needs no fitting - it is just the ratio of the two image sizes, which are both on disk. It does vary a lot *between* plans, so it must be computed per plan and never assumed.

Measured over 60 sampled plans:

- scaled/original ratio ranges **0.40x - 5.05x**
- max |x ratio - y ratio| within a plan: **0.0030** (0 would be perfectly uniform)

For contrast, the `model.svg` annotations have per-axis ratios that disagree with each other (0.909/1.002, 0.949/1.065, 1.025/1.054), which is why SVG polygons are not drawn onto the page anywhere in this project.

## Per-image sizes (sample)

| Plan | COCO declares | F1_original.png | F1_scaled.png | Matches |
| --- | --- | --- | --- | --- |
| 13500 | 575x774 | 575x774 | 778x1047 | F1_original.png |
| 1636 | 865x530 | 865x530 | 1699x1041 | F1_original.png |
| 8731 | 680x482 | 680x482 | 1393x987 | F1_original.png |
| 1432 | 852x569 | 852x569 | 1378x920 | F1_original.png |
| 654 | 1444x570 | 1444x570 | 2937x1159 | F1_original.png |
| 6726 | 863x838 | 863x838 | 1513x1469 | F1_original.png |
| 12214 | 408x466 | 408x466 | 984x1124 | F1_original.png |
| 4553 | 426x628 | 426x628 | 1268x1869 | F1_original.png |
| 2530 | 851x605 | 851x605 | 2220x1578 | F1_original.png |
| 6596 | 196x494 | 196x494 | 989x2494 | F1_original.png |
| 1116 | 940x606 | 940x606 | 2443x1575 | F1_original.png |
| 12139 | 816x1709 | 816x1709 | 745x1560 | F1_original.png |
| 1838 | 429x649 | 429x649 | 581x880 | F1_original.png |
| 1141 | 993x488 | 993x488 | 1330x654 | F1_original.png |
| 12002 | 979x621 | 979x621 | 1495x948 | F1_original.png |
| 11565 | 1916x1424 | 1916x1424 | 1452x1079 | F1_original.png |
| 3984 | 1869x699 | 1869x699 | 2477x926 | F1_original.png |
| 8380 | 493x799 | 493x799 | 828x1342 | F1_original.png |
| 774 | 1800x900 | 1800x900 | 2096x1048 | F1_original.png |
| 20003 | 1094x744 | 1094x744 | 1439x979 | F1_original.png |
| 3162 | 565x798 | 565x798 | 1008x1423 | F1_original.png |
| 6007 | 972x508 | 972x508 | 2210x1155 | F1_original.png |
| 1982 | 633x899 | 633x899 | 2867x4072 | F1_original.png |
| 7780 | 919x694 | 919x694 | 1886x1424 | F1_original.png |
| 19 | 1060x556 | 1060x556 | 2566x1346 | F1_original.png |
| 4617 | 341x525 | 341x525 | 813x1252 | F1_original.png |
| 9488 | 538x367 | 538x367 | 2327x1587 | F1_original.png |
| 6152 | 710x690 | 710x690 | 1887x1834 | F1_original.png |
| 12871 | 1060x1074 | 1060x1074 | 2350x2381 | F1_original.png |
| 7715 | 2564x2064 | 2564x2064 | 1024x824 | F1_original.png |
| 20028 | 1056x480 | 1056x480 | 2006x912 | F1_original.png |
| 9198 | 573x802 | 573x802 | 1548x2166 | F1_original.png |
| 2539 | 680x731 | 680x731 | 1239x1332 | F1_original.png |
| 3516 | 767x797 | 767x797 | 726x755 | F1_original.png |
| 6521 | 834x1299 | 834x1299 | 981x1529 | F1_original.png |
| 1191 | 883x592 | 883x592 | 2319x1555 | F1_original.png |
| 13685 | 1763x917 | 1763x917 | 2271x1181 | F1_original.png |
| 11750 | 960x1280 | 960x1280 | 1267x1689 | F1_original.png |
| 6123 | 784x716 | 784x716 | 2079x1899 | F1_original.png |
| 6303 | 453x592 | 453x592 | 909x1187 | F1_original.png |
