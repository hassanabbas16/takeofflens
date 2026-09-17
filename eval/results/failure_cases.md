# Failure cases

Top 3 worst plans per approach, from `tier3_sample50.json`. Reasons are mechanical, not narrative.


## `rules`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9976` | 9/20 | 0/0 | 11 area(s) not corroborated by the reference |
| `5046` | 12/18 | 0/0 | 10 area(s) not corroborated by the reference |
| `3984` | 9/14 | 0/0 | 8 area(s) not corroborated by the reference |

## `ocr+llm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `5046` | 12/18 | 0/0 | 10 area(s) not corroborated by the reference |
| `3984` | 8/14 | 0/0 | 8 area(s) not corroborated by the reference; 1 fewer labels than free rules baseline |
| `5886` | 7/13 | 0/0 | 1 hallucination(s); 6 area(s) not corroborated by the reference |

## `vlm`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9976` | 10/20 | 0/0 | 17 area(s) not corroborated by the reference |
| `5046` | 9/18 | 0/0 | 15 area(s) not corroborated by the reference; 3 fewer labels than free rules baseline |
| `3984` | 10/14 | 0/0 | 10 area(s) not corroborated by the reference |

## `hybrid`

| Plan | Labels | Areas | What went wrong |
| --- | --- | --- | --- |
| `9270` | 8/10 | 0/0 | 5 hallucination(s); 6 area(s) not corroborated by the reference |
| `9976` | 10/20 | 0/0 | 14 area(s) not corroborated by the reference |
| `5046` | 12/18 | 0/0 | 1 hallucination(s); 11 area(s) not corroborated by the reference |