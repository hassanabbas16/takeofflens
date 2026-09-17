# Tier 3 cost probe

- Plans: 50 `high_quality_architectural` test plans
- Text model: `claude-haiku-4-5`
- Vision model: `claude-haiku-4-5`
- Pricing: https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-09-17, standard mode
- Reference: 495 labels, 15 areas actually printed (736 rooms in the SVG annotation)
- **Area ground truth covers 2 of 50 plans** (hand-verified, `printed_areas_6.json`). Label ground truth is automatic (Tier 2, from `model.svg`) and covers all of them.

> **Read the Areas and Spurious columns with care.** On the 48 plans with no hand-verified area ground truth the printed-area count defaults to 0, which means *not checked*, not *verified none*. Areas found on those plans are therefore counted as spurious whether or not the drawing prints them. Labels, hallucinations and cost are unaffected. A real area/dimension accuracy number needs the Tier 4 gold set.


| Approach | Labels | Areas (printed) | Spurious areas | Hallucinations | Mean latency | Mean cost/page | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rules | 246/495 | 3/15 | 70 | 0 | 0.0s | $0.00000 | $0.0000 |
| ocr+llm | 234/495 | 3/15 | 83 | 2 | 0.0s | $0.00299 | $0.1496 |
| vlm | 292/495 | 0/15 | 96 | 0 | 0.0s | $0.00368 | $0.1841 |
| hybrid | 289/495 | 3/15 | 106 | 10 | 0.0s | $0.00415 | $0.2076 |

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
