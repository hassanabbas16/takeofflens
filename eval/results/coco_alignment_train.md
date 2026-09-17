# Do the supplied COCO boxes line up with the page images?

Checked before Phase 8, because the `model.svg` annotations in this same dataset do **not** map to the page by any constant. An assumption about coordinate space is worth nothing until it is measured.

- Split: `train` - 4200 images, 173023 annotations, categories `['room', 'wall']`
- Sample: 40 images, seed 20260917
- Plans whose image could not be found on the mount: 0

## 1. Which image do the declared sizes match?

| Matches | Images |
| --- | --- |
| `F1_original.png` | 40 |

## 2. Containment

- Boxes inside the declared frame: **1597/1597** (100.0%)
- Mean page area covered by the union of `room` boxes: **51.2%**

## 3. Ink overlap

Measured on `wall` boxes, which should be almost solid ink. The control is the same box shifted by a third of the page - what the real boxes would look like if the annotations were in the wrong coordinate space.

- Mean dark-pixel fraction inside `wall` boxes: **0.608**
- Same boxes, shifted (control):               **0.117**
- Ratio: **5.18x**

Room boxes are deliberately not used for this: a room is mostly white by definition, so a correct box and a misplaced one score alike (the first version of this check scored 1.19x and proved nothing). Instead, for rooms:

- Ink in a room box's outer band (its walls): **0.251**
- Ink in its interior (the floor):            **0.112**
- Ratio: **2.23x**

## Verdict

**Aligned.** The annotations are in `F1_original.png` pixel space.

Sample renders with boxes drawn: `/eval/labelling/coco_check` - look at them before trusting this.

## Per-image sizes (sample)

| Plan | COCO declares | F1_original.png | F1_scaled.png | Matches |
| --- | --- | --- | --- | --- |
| 2195 | 646x391 | 646x391 | 1138x689 | F1_original.png |
| 12780 | 3307x2338 | 3307x2338 | 3200x2262 | F1_original.png |
| 11512 | 856x492 | 856x492 | 1457x837 | F1_original.png |
| 11032 | 1024x743 | 1024x743 | 2104x1527 | F1_original.png |
| 7885 | 713x527 | 713x527 | 1747x1291 | F1_original.png |
| 9434 | 604x562 | 604x562 | 1841x1713 | F1_original.png |
| 14498 | 1007x638 | 1007x638 | 1392x882 | F1_original.png |
| 13239 | 1542x938 | 1542x938 | 2106x1281 | F1_original.png |
| 10311 | 328x478 | 328x478 | 1052x1533 | F1_original.png |
| 10420 | 580x477 | 580x477 | 1502x1235 | F1_original.png |
| 11377 | 362x580 | 362x580 | 1567x2510 | F1_original.png |
| 4006 | 253x533 | 253x533 | 466x982 | F1_original.png |
| 6334 | 942x596 | 942x596 | 1421x899 | F1_original.png |
| 14529 | 1163x902 | 1163x902 | 1604x1244 | F1_original.png |
| 4766 | 776x639 | 776x639 | 1972x1624 | F1_original.png |
| 1859 | 587x744 | 587x744 | 915x1159 | F1_original.png |
| 3849 | 590x938 | 590x938 | 746x1187 | F1_original.png |
| 12219 | 559x353 | 559x353 | 1481x936 | F1_original.png |
| 8129 | 518x327 | 518x327 | 1384x874 | F1_original.png |
| 13067 | 373x501 | 373x501 | 1023x1374 | F1_original.png |
| 1171 | 4960x3507 | 4960x3507 | 2053x1452 | F1_original.png |
| 12940 | 2338x1700 | 2338x1700 | 1832x1332 | F1_original.png |
| 4 | 1803x845 | 1803x845 | 1606x753 | F1_original.png |
| 13645 | 480x731 | 480x731 | 789x1201 | F1_original.png |
| 11380 | 746x757 | 746x757 | 1167x1184 | F1_original.png |
| 10557 | 1255x895 | 1255x895 | 2603x1857 | F1_original.png |
| 10182 | 534x551 | 534x551 | 819x845 | F1_original.png |
| 6429 | 453x644 | 453x644 | 884x1257 | F1_original.png |
| 7713 | 1163x923 | 1163x923 | 1370x1087 | F1_original.png |
| 10156 | 696x736 | 696x736 | 1438x1521 | F1_original.png |
| 8127 | 287x393 | 287x393 | 898x1230 | F1_original.png |
| 14797 | 337x650 | 337x650 | 704x1358 | F1_original.png |
| 13603 | 799x433 | 799x433 | 1330x721 | F1_original.png |
| 11127 | 2880x3128 | 2880x3128 | 2419x2627 | F1_original.png |
| 10436 | 1056x768 | 1056x768 | 2151x1564 | F1_original.png |
| 12879 | 512x806 | 512x806 | 652x1027 | F1_original.png |
| 12665 | 772x617 | 772x617 | 1811x1447 | F1_original.png |
| 8792 | 3008x2944 | 3008x2944 | 3110x3044 | F1_original.png |
| 12775 | 2479x3504 | 2479x3504 | 1621x2291 | F1_original.png |
| 7870 | 1155x679 | 1155x679 | 3032x1783 | F1_original.png |
