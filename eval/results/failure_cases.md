# Failure cases

Top 5 worst plans per approach, from `tier3_sample50.json`. Reasons are mechanical, not narrative.


## `rules`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `5046` | 12/18 | 0/0 | 10 area(s) on a plan with no verified area ground truth |
| `3984` | 9/14 | 0/0 | 8 area(s) on a plan with no verified area ground truth |
| `5974` | 3/11 | 0/0 | only 3/11 labels; 4 area(s) on a plan with no verified area ground truth |
| `20000` | 0/1 | 0/0 | found none of the 1 reference labels; 1 area(s) on a plan with no verified area ground truth |
| `366` | 11/14 | 0/0 | 5 area(s) on a plan with no verified area ground truth |

## `ocr+llm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `1116` | 6/15 | 0/0 | 8 hallucination(s); 6 area(s) on a plan with no verified area ground truth |
| `5886` | 7/13 | 0/0 | 5 hallucination(s); 6 area(s) on a plan with no verified area ground truth |
| `416` | 8/11 | 0/0 | 5 hallucination(s); 5 area(s) on a plan with no verified area ground truth |
| `9270` | 5/10 | 0/0 | 5 hallucination(s); 4 area(s) on a plan with no verified area ground truth; 1 fewer labels than free rules baseline |
| `9976` | 10/20 | 0/0 | 4 hallucination(s); 6 area(s) on a plan with no verified area ground truth |

## `vlm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9976` | 10/20 | 0/0 | 17 area(s) on a plan with no verified area ground truth |
| `5046` | 9/18 | 0/0 | 15 area(s) on a plan with no verified area ground truth; 3 fewer labels than free rules baseline |
| `3984` | 10/14 | 0/0 | 10 area(s) on a plan with no verified area ground truth |
| `6704` | 9/14 | 0/0 | 9 area(s) on a plan with no verified area ground truth |
| `5570` | 3/9 | 0/0 | only 3/9 labels; 5 area(s) on a plan with no verified area ground truth; 1 fewer labels than free rules baseline |

## `hybrid`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9976` | 10/20 | 0/0 | 11 hallucination(s); 14 area(s) on a plan with no verified area ground truth |
| `9270` | 8/10 | 0/0 | 10 hallucination(s); 6 area(s) on a plan with no verified area ground truth |
| `416` | 10/11 | 0/0 | 6 hallucination(s); 6 area(s) on a plan with no verified area ground truth |
| `1116` | 9/15 | 0/0 | 6 hallucination(s); 5 area(s) on a plan with no verified area ground truth |
| `5886` | 7/13 | 0/0 | 5 hallucination(s); 7 area(s) on a plan with no verified area ground truth |