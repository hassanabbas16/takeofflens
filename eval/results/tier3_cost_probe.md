# Tier 3 cost probe

- Plans: 6 `high_quality_architectural` test plans
- Text model: `claude-haiku-4-5`
- Vision model: `claude-haiku-4-5`
- Pricing: https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-09-17, standard mode
- Reference: 62 labels, 29 areas actually printed (89 rooms in the SVG annotation)

| Approach | Labels | Areas (printed) | Spurious areas | Hallucinations | Mean latency | Mean cost/page | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ocr+llm | 31/62 | 7/29 | 3 | 2 | 12.4s | $0.00618 | $0.0371 |
| vlm | 37/62 | 3/29 | 0 | 2 | 8.6s | $0.00749 | $0.0449 |
| hybrid | 32/62 | 8/29 | 3 | 2 | 18.3s | $0.00882 | $0.0529 |

## Extrapolation

Cost of the full `high_quality_architectural` test split (270 plans), at the measured per-page rate:

| Approach | Projected cost |
| --- | --- |
| ocr+llm | $1.67 |
| vlm | $2.02 |
| hybrid | $2.38 |

**STOP.** This is the cost gate. The full run needs explicit approval.
