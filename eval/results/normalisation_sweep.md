# OCR area-unit repair: what it buys

- 50 `high_quality_architectural` test plans (the Tier 3 sample), 2403 OCR tokens
- Free: `rules` is deterministic and recomputed from the OCR cache in every arm.

> **The `rules` row is the only one that moves.** The `ocr+llm`, `vlm` and `hybrid`
> numbers in `tier3_sample50.md` were produced against **pre-repair** prompts and
> are not re-run here - that would cost money. They are therefore not comparable to
> a post-repair `rules` on the pairing that fed them, and the results table says so
> next to the figures.

| Arm | Area-readable tokens | `rules` rooms | `rules` areas | Fabricated areas |
| --- | --- | --- | --- | --- |
| `off` | 158 | 433 | 161 | 0 |
| `on` | 200 | 433 | 188 | 0 |
| `aggressive` | 215 | 439 | 197 | 0 |

Repair moves area-readable tokens **158 -> 200** (+42) and `rules` areas **161 -> 188** (+27), changing the output on 8 of 50 plans.

The fabrication column is over the 2 sampled plans that have hand-verified printed-area ground truth, counting areas reported on a plan the human recorded as printing none. It must not rise.

## Why the aggressive arm is not shipped

It adds 9 further `rules` areas over `on`, by assuming a bare trailing `m` means `m2`. That is a guess about a token that could legitimately be a length, not a repair of a known recogniser defect, and it is the kind of assumption this project reports rather than makes.

## Repairs that fired

- `6,7 m^{2}$` -> `6,7 m2` = 6.7 m2
- `6,0 m^{2}$` -> `6,0 m2` = 6 m2
- `$12,3 m^{}$` -> `12,3 m2` = 12.3 m2
- `$3,2 m^{2}$` -> `3,2 m2` = 3.2 m2
- `2,3 m{2` -> `2,3 m2` = 2.3 m2
- `$8,1 m^{2}$` -> `8,1 m2` = 8.1 m2
- `9,5m^2}$` -> `9,5m2` = 9.5 m2
- `$10.0 $m^{}$` -> `10.0 m2` = 10 m2
- `15.5 m^2}$` -> `15.5 m2` = 15.5 m2
- `34.0 m^{}$` -> `34.0 m2` = 34 m2
