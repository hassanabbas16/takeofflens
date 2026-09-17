# Do the supplied COCO boxes line up with the page images?

Checked before Phase 8, because the `model.svg` annotations in this same dataset do **not** map to the page by any constant. An assumption about coordinate space is worth nothing until it is measured.

- Split: `val` - 400 images, 15926 annotations, categories `['room', 'wall']`
- Sample: 40 images, seed 20260917
- Plans whose image could not be found on the mount: 0

## 1. Which image do the declared sizes match?

| Matches | Images |
| --- | --- |
| `F1_original.png` | 40 |

## 2. Containment

- Boxes inside the declared frame: **1530/1530** (100.0%)
- Mean page area covered by the union of `room` boxes: **52.9%**

## 3. Ink overlap

Measured on `wall` boxes, which should be almost solid ink. The control is the same box shifted by a third of the page - what the real boxes would look like if the annotations were in the wrong coordinate space.

- Mean dark-pixel fraction inside `wall` boxes: **0.631**
- Same boxes, shifted (control):               **0.120**
- Ratio: **5.28x**

Room boxes are deliberately not used for this: a room is mostly white by definition, so a correct box and a misplaced one score alike (the first version of this check scored 1.19x and proved nothing). Instead, for rooms:

- Ink in a room box's outer band (its walls): **0.232**
- Ink in its interior (the floor):            **0.120**
- Ratio: **1.93x**

## Verdict

**Aligned.** The annotations are in `F1_original.png` pixel space.

Sample renders with boxes drawn: `/eval/labelling/coco_check` - look at them before trusting this.

## Per-image sizes (sample)

| Plan | COCO declares | F1_original.png | F1_scaled.png | Matches |
| --- | --- | --- | --- | --- |
| 7046 | 640x410 | 640x410 | 1178x755 | F1_original.png |
| 3195 | 1600x868 | 1600x868 | 2947x1599 | F1_original.png |
| 1647 | 488x490 | 488x490 | 955x959 | F1_original.png |
| 20018 | 732x611 | 732x611 | 2047x1709 | F1_original.png |
| 4864 | 946x817 | 946x817 | 1542x1332 | F1_original.png |
| 9157 | 628x1318 | 628x1318 | 817x1715 | F1_original.png |
| 13804 | 844x1529 | 844x1529 | 887x1608 | F1_original.png |
| 9936 | 877x797 | 877x797 | 1945x1768 | F1_original.png |
| 5559 | 1238x972 | 1238x972 | 1848x1451 | F1_original.png |
| 1967 | 1427x719 | 1427x719 | 2497x1258 | F1_original.png |
| 9860 | 649x439 | 649x439 | 1514x1024 | F1_original.png |
| 14255 | 543x605 | 543x605 | 1130x1259 | F1_original.png |
| 6514 | 768x785 | 768x785 | 932x953 | F1_original.png |
| 6301 | 464x676 | 464x676 | 932x1358 | F1_original.png |
| 14245 | 567x346 | 567x346 | 1259x769 | F1_original.png |
| 14030 | 496x454 | 496x454 | 1103x1009 | F1_original.png |
| 1361 | 542x408 | 542x408 | 1376x1036 | F1_original.png |
| 751 | 1310x799 | 1310x799 | 1620x988 | F1_original.png |
| 2572 | 862x1057 | 862x1057 | 2283x2799 | F1_original.png |
| 1554 | 827x423 | 827x423 | 1530x782 | F1_original.png |
| 4548 | 1023x655 | 1023x655 | 1998x1279 | F1_original.png |
| 6286 | 442x757 | 442x757 | 773x1324 | F1_original.png |
| 1611 | 297x291 | 297x291 | 780x764 | F1_original.png |
| 7550 | 275x340 | 275x340 | 791x978 | F1_original.png |
| 3828 | 454x505 | 454x505 | 1134x1262 | F1_original.png |
| 6632 | 1033x877 | 1033x877 | 1760x1495 | F1_original.png |
| 1089 | 494x722 | 494x722 | 812x1187 | F1_original.png |
| 7519 | 1924x824 | 1924x824 | 3100x1328 | F1_original.png |
| 14565 | 423x851 | 423x851 | 852x1714 | F1_original.png |
| 4542 | 515x428 | 515x428 | 872x725 | F1_original.png |
| 9436 | 968x929 | 968x929 | 1459x1400 | F1_original.png |
| 1061 | 1337x931 | 1337x931 | 2015x1403 | F1_original.png |
| 6257 | 507x440 | 507x440 | 1254x1088 | F1_original.png |
| 6082 | 655x448 | 655x448 | 1019x697 | F1_original.png |
| 5682 | 773x830 | 773x830 | 1300x1396 | F1_original.png |
| 333 | 1319x619 | 1319x619 | 1319x619 | F1_original.png |
| 9939 | 442x851 | 442x851 | 808x1557 | F1_original.png |
| 14054 | 849x1065 | 849x1065 | 1452x1822 | F1_original.png |
| 2567 | 1412x1606 | 1412x1606 | 3147x3579 | F1_original.png |
| 7864 | 731x392 | 731x392 | 1312x704 | F1_original.png |
