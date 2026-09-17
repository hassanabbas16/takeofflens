# Hallucination classification - 50-plan Tier 3 run

Replayed from the raw batch output. Retrieving a finished batch's results is free and they are retained for 29 days, so this cost nothing.

`vlm` is absent because it grounds softly by design - the vision model may read an area OCR missed, so a disagreement with OCR would measure OCR - and it reported no hallucinations.

## Summary

| Approach | Flagged before | Real after the fix | Removed as artifacts |
| --- | --- | --- | --- |
| `ocr+llm` | 39 | 2 | 37 |
| `hybrid` | 57 | 10 | 47 |

## `ocr+llm` - 39 flagged by the original checker

| Category | Count | Share |
| --- | --- | --- |
| `scoring_artifact:area_in_garbled_token` | 37 | 95% |
| `invented:area_not_on_page` | 2 | 5% |

**`scoring_artifact:area_in_garbled_token`**

- plan 1041: "OH." area 16 m2 is in token(s) ['OH. 16.0 n?'] that the parser cannot read as an area

**`invented:area_not_on_page`**

- plan 20003: "LH+K" area 85 m2 appears nowhere in the OCR text

## `hybrid` - 57 flagged by the original checker

| Category | Count | Share |
| --- | --- | --- |
| `scoring_artifact:area_in_garbled_token` | 47 | 82% |
| `invented:area_not_on_page` | 10 | 18% |

**`scoring_artifact:area_in_garbled_token`**

- plan 1041: "OH" area 16 m2 is in token(s) ['OH. 16.0 n?'] that the parser cannot read as an area

**`invented:area_not_on_page`**

- plan 3984: "oH" area 18 m2 appears nowhere in the OCR text
