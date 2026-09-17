# Failure cases

Top 2 worst plans per approach, from `tier3_sample50.json`. Reasons are mechanical, not narrative.


## `rules`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `5046` | 12/18 | 0/0 | 10 area(s) on a plan with no verified area ground truth |
| `3984` | 9/14 | 0/0 | 8 area(s) on a plan with no verified area ground truth |

## `ocr+llm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `5046` | 12/18 | 0/0 | 10 area(s) on a plan with no verified area ground truth |
| `3984` | 8/14 | 0/0 | 8 area(s) on a plan with no verified area ground truth; 1 fewer labels than free rules baseline |

## `vlm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9976` | 10/20 | 0/0 | 17 area(s) on a plan with no verified area ground truth |
| `5046` | 9/18 | 0/0 | 15 area(s) on a plan with no verified area ground truth; 3 fewer labels than free rules baseline |

## `hybrid`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9270` | 8/10 | 0/0 | 5 hallucination(s); 6 area(s) on a plan with no verified area ground truth |
| `9976` | 10/20 | 0/0 | 14 area(s) on a plan with no verified area ground truth |