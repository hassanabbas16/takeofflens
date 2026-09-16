# Tier 3 cost probe

- Plans: 6 `high_quality_architectural` test plans
- Text model: `claude-haiku-4-5`
- Vision model: `claude-haiku-4-5`
- Pricing: https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-09-17, standard mode
- Reference: 62 labels, 29 areas actually printed (89 rooms in the SVG annotation)

| Approach | Labels | Areas (printed) | Spurious areas | Hallucinations | Mean latency | Mean cost/page | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rules | 36/62 | 7/29 | 4 | 0 | 0.0s | $0.00000 | $0.0000 |
| ocr+llm | 33/62 | 7/29 | 3 | 0 | 12.1s | $0.00607 | $0.0364 |
| vlm | 39/62 | 3/29 | 0 | 0 | 8.5s | $0.00758 | $0.0455 |
| hybrid | 41/62 | 9/29 | 3 | 0 | 10.1s | $0.00884 | $0.0531 |

## Extrapolation

Cost of the full `high_quality_architectural` test split (270 plans), at the measured per-page rate:

| Approach | Projected cost |
| --- | --- |
| rules | $0.00 |
| ocr+llm | $1.64 |
| vlm | $2.05 |
| hybrid | $2.39 |

Measured spend this run: **$0.1350** (cap $0.30)


**STOP.** This is the cost gate. The full run needs explicit approval.
