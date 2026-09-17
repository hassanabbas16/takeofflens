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
- plan 1041: "KEITTIO" area 9.5 m2 is in token(s) ['9,5m^2}$'] that the parser cannot read as an area
- plan 1116: "OLOH. 17.6m" area 17.6 m2 is in token(s) ['OLOH. 17.6m'] that the parser cannot read as an area
- plan 1116: "MH 8,0m" area 8 m2 is in token(s) ['MH 8,0m', 'MH 8,0m'] that the parser cannot read as an area
- plan 1116: "MH18,4m" area 18.4 m2 is in token(s) ['MH18,4m'] that the parser cannot read as an area

**`invented:area_not_on_page`**

- plan 20003: "LH+K" area 85 m2 appears nowhere in the OCR text
- plan 5886: "MH" area 8.3 m2 appears nowhere in the OCR text

## `hybrid` - 57 flagged by the original checker

| Category | Count | Share |
| --- | --- | --- |
| `scoring_artifact:area_in_garbled_token` | 47 | 82% |
| `invented:area_not_on_page` | 10 | 18% |

**`scoring_artifact:area_in_garbled_token`**

- plan 1041: "OH" area 16 m2 is in token(s) ['OH. 16.0 n?'] that the parser cannot read as an area
- plan 1041: "KEITTIO" area 9.5 m2 is in token(s) ['9,5m^2}$'] that the parser cannot read as an area
- plan 1116: "OLOHUONE" area 17.6 m2 is in token(s) ['OLOH. 17.6m'] that the parser cannot read as an area
- plan 1116: "MH" area 8 m2 is in token(s) ['MH 8,0m', 'MH 8,0m'] that the parser cannot read as an area
- plan 1116: "MH" area 18.4 m2 is in token(s) ['MH18,4m'] that the parser cannot read as an area

**`invented:area_not_on_page`**

- plan 3984: "oH" area 18 m2 appears nowhere in the OCR text
- plan 5046: "TYGH" area 8.9 m2 appears nowhere in the OCR text
- plan 5886: "MH" area 9.3 m2 appears nowhere in the OCR text
- plan 7809: "WC/PH" area 4.5 m2 appears nowhere in the OCR text
- plan 7809: "VAR" area 4 m2 appears nowhere in the OCR text
