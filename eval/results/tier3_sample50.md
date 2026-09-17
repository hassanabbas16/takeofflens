# Tier 3 cost probe

- Plans: 50 `high_quality_architectural` test plans
- Text model: `claude-haiku-4-5`
- Vision model: `claude-haiku-4-5`
- Pricing: https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-09-17, standard mode
- Reference: 495 name labels in `model.svg` (736 annotated rooms)
- Label ground truth is automatic, from the `model.svg` name labels, and covers every plan.

> **This table does not score areas.** `Areas reported` is a raw count with no ground truth behind it, included so a change in behaviour is visible. Area accuracy is measured separately against the `model.svg` polygons over all 50 plans - see `tier2_area_accuracy.md`.


> **`rules` is post-repair; the three paid rows are pre-repair.** `rules` is free and deterministic, so it is recomputed here and includes the OCR area-unit repair (`normalise_area_unit`). The `ocr+llm`, `vlm` and `hybrid` outputs are replayed from cache and were produced against prompts built **before** that repair existed; re-running them would cost money and has not been done. Their label, cost and hallucination figures are unaffected by the repair, but any `rules`-vs-paid comparison on **areas** is not like-for-like. See `normalisation_sweep.md` for what the repair changes.


| Approach | Labels | Areas reported | Hallucinations | Mean latency | Mean cost/page | Total |
| --- | --- | --- | --- | --- | --- | --- |
| rules | 246/495 | 188 | 0 | 0.0s | $0.00000 | $0.0000 |
| ocr+llm | 234/495 | 177 | 2 | 0.0s | $0.00299 | $0.1496 |
| vlm | 292/495 | 184 | 0 | 0.0s | $0.00368 | $0.1841 |
| hybrid | 289/495 | 210 | 10 | 0.0s | $0.00415 | $0.2076 |

## Extrapolation

Cost of the full `high_quality_architectural` test split (270 plans), at the measured per-page rate:

| Approach | Projected cost |
| --- | --- |
| rules | $0.00 |
| ocr+llm | $0.81 |
| vlm | $0.99 |
| hybrid | $1.12 |

Cost of the results in this table: **$0.5413**

New spend this run: **$0.0000** (cap $0.73)


**STOP.** This is the cost gate. The full run needs explicit approval.
