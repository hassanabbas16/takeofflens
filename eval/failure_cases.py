"""Surface the worst per-plan results for each approach, with plan ids.

Aggregate scores hide which plans broke and why. This ranks each approach's plans by how
badly they went and prints the top few with their ids, so a failure can be opened and looked
at rather than argued about.

A failure here is one of:

- the call failed outright (no result at all)
- the approach found none of the reference labels on a plan that has them
- it reported areas on a plan with no hand-verified printed-area ground truth. Treat this
  as a *flag, not a verdict*: the printed-area count is only known for the plans in
  eval/ground_truth/printed_areas_6.json, and defaults to 0 everywhere else, so "0 printed"
  usually means "nobody checked". It ranks plans worth looking at by eye, and only the
  plans covered by that file can be called fabrications outright.
- it produced hallucinations (a cited token id that does not exist, or an unsupported area)
- it did clearly worse than the free `rules` baseline on the same plan

Reads the results JSON. No API calls, no cost.

    python eval/failure_cases.py --results /eval/results/tier3_sample50.json --top 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def severity(row: dict, rules_row: dict | None) -> tuple[int, str]:
    """Rank a plan's result. Higher is worse; the reason is what gets printed."""
    if not row.get("ok"):
        return 100, f"call failed: {row.get('error')}"

    ref_labels = row.get("ref_labels", 0)
    labels = row.get("labels", 0)
    spurious = row.get("spurious_areas", 0)
    halluc = row.get("hallucinations", 0)
    printed = row.get("printed_areas", 0)
    areas = row.get("areas_on_printing_plans", 0)

    reasons: list[str] = []
    score = 0

    if ref_labels and labels == 0:
        score += 50
        reasons.append(f"found none of the {ref_labels} reference labels")
    elif ref_labels and labels / ref_labels < 0.34:
        score += 25
        reasons.append(f"only {labels}/{ref_labels} labels")

    if halluc:
        score += 20 * halluc
        reasons.append(f"{halluc} hallucination(s)")

    if spurious:
        # Weighted low, and worded as unverified, because for most plans the zero
        # denominator means "not checked" rather than "verified to print none". Calling
        # these fabrications would be an accuracy claim the ground truth cannot support.
        score += 10 * spurious
        reasons.append(
            f"{spurious} area(s) on a plan with no verified area ground truth"
        )

    if printed and areas == 0:
        score += 15
        reasons.append(f"missed all {printed} printed areas")

    if rules_row and rules_row.get("ok"):
        deficit = rules_row.get("labels", 0) - labels
        if deficit > 0:
            score += 5 * deficit
            reasons.append(f"{deficit} fewer labels than free rules baseline")

    return score, "; ".join(reasons) if reasons else "no issue"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path("/eval/results/tier3_sample50.json"))
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=Path("/eval/results/failure_cases.md"))
    args = parser.parse_args()

    data = load(args.results)
    results = data["results"]
    rules_by_plan = {r["plan"]: r for r in results.get("rules", [])}

    lines = ["# Failure cases\n",
             f"Top {args.top} worst plans per approach, from "
             f"`{args.results.name}`. Reasons are mechanical, not narrative.\n"]

    for approach, rows in results.items():
        ranked = sorted(
            ((severity(r, rules_by_plan.get(r["plan"])), r) for r in rows),
            key=lambda pair: -pair[0][0],
        )
        worst = [(sev, r) for (sev, r) in ranked if sev[0] > 0][: args.top]

        print(f"\n=== {approach} ===")
        lines.append(f"\n## `{approach}`\n")
        if not worst:
            print("  no plans scored as failures")
            lines.append("No plans scored as failures.\n")
            continue

        lines.append("| Plan | Labels | Areas | What went wrong |")
        lines.append("| --- | --- | --- | --- |")
        for (score, reason), row in worst:
            labels = (f"{row.get('labels', 0)}/{row.get('ref_labels', 0)}"
                      if row.get("ok") else "-")
            areas = (f"{row.get('areas_on_printing_plans', 0)}/{row.get('printed_areas', 0)}"
                     if row.get("ok") else "-")
            print(f"  {row['plan']:>6}  labels={labels:<8} areas={areas:<7} {reason}")
            lines.append(f"| `{row['plan']}` | {labels} | {areas} | {reason} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
