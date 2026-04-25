"""Stage 3: aggregate classified predictions into tables, examples, report.

Reads `classified_predictions.jsonl`. Emits:
  - by_model.csv     : rows=models    cols=CATEGORY_CODES + total
  - by_condition.csv : rows=conditions
  - by_scope.csv     : rows=scope labels
  - by_hop.csv       : rows=hop counts (1..4)
  - examples.md      : one section per category, with the prediction
                      whose F1 is closest to the median within that bucket.
  - report.md        : headline counts + links to the above. κ section
                      placeholder is filled later by `kappa_sample.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.taxonomy import CATEGORY_CODES

AXES = ["by_model", "by_condition", "by_scope", "by_hop"]
AXIS_FIELD = {"by_model": "model", "by_condition": "condition",
              "by_scope": "scope", "by_hop": "hop_count"}


def _load(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _axis_matrix(rows: list[dict], field: str) -> dict[str, Counter]:
    """{axis_value: Counter over category codes}."""
    grid: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        key = r.get(field)
        if key is None:
            continue
        key = str(key)
        cause = r.get("primary_cause") or "unlabeled"
        grid[key][cause] += 1
    return grid


def _write_axis_csv(grid: dict[str, Counter], field: str, out: Path) -> None:
    cats = CATEGORY_CODES + ["unlabeled"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([field] + cats + ["total"])
        for key in sorted(grid):
            counts = grid[key]
            total = sum(counts.values())
            w.writerow([key] + [counts.get(c, 0) for c in cats] + [total])


def _pick_example(rows: list[dict], cause: str) -> dict | None:
    """Return the row whose f1 is closest to the median of its bucket.

    Returns None if the bucket is empty.
    """
    bucket = [r for r in rows if r.get("primary_cause") == cause]
    if not bucket:
        return None
    f1s = [float(r.get("f1") or 0.0) for r in bucket]
    med = statistics.median(f1s)
    return min(bucket, key=lambda r: abs(float(r.get("f1") or 0.0) - med))


def _examples_md(rows: list[dict]) -> str:
    lines: list[str] = ["# Failure Examples", "",
                        "One representative prediction per category "
                        "(closest to the median F1 within that bucket).",
                        ""]
    for c in CATEGORY_CODES:
        ex = _pick_example(rows, c)
        if ex is None:
            continue
        lines.append(f"## {c}")
        lines.append("")
        lines.append(f"- **Question:** {ex.get('question')}")
        lines.append(f"- **Gold:** {ex.get('gold')}")
        lines.append(f"- **Prediction:** `{ex.get('prediction')}`")
        lines.append(f"- **F1:** {ex.get('f1')}")
        lines.append(f"- **Scope / hop:** {ex.get('scope')} / {ex.get('hop_count')}")
        lines.append(f"- **Condition × model:** {ex.get('condition')} / {ex.get('model')}")
        reason = ex.get("reason")
        if reason:
            lines.append(f"- **Classifier rationale:** {reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(rows: list[dict]) -> str:
    n = len(rows)
    counts = Counter(r.get("primary_cause") or "unlabeled" for r in rows)
    nf = counts.get("NF", 0)
    ambiguous_left = counts.get("unlabeled", 0)
    lines = [
        "# Failure Taxonomy — Aggregate Report",
        "",
        f"Total classified predictions: **{n}**",
        f"Non-failures (NF): **{nf}** ({100*nf/max(n,1):.1f}%)",
        f"Unlabeled remaining: **{ambiguous_left}** (expected 0 after Stage 2)",
        "",
        "## Headline counts by category",
        "",
        "| Code | Count | % |",
        "|------|------:|---:|",
    ]
    for c in CATEGORY_CODES + ["unlabeled"]:
        n_c = counts.get(c, 0)
        lines.append(f"| {c} | {n_c} | {100*n_c/max(n,1):.1f}% |")
    lines += [
        "",
        "## Tables",
        "",
        "- [by_model.csv](by_model.csv)",
        "- [by_condition.csv](by_condition.csv)",
        "- [by_scope.csv](by_scope.csv)",
        "- [by_hop.csv](by_hop.csv)",
        "- [examples.md](examples.md)",
        "",
        "## Reliability",
        "",
        "(Filled in by `scripts/kappa_sample.py`.)",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--out-dir",
                        default="data/eval/failure_taxonomy")
    args = parser.parse_args()

    rows = _load(Path(args.classified))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ax in AXES:
        grid = _axis_matrix(rows, AXIS_FIELD[ax])
        _write_axis_csv(grid, AXIS_FIELD[ax], out_dir / f"{ax}.csv")

    (out_dir / "examples.md").write_text(_examples_md(rows), encoding="utf-8")
    (out_dir / "report.md").write_text(_report_md(rows), encoding="utf-8")
    print(f"Wrote {len(AXES)} CSVs + examples.md + report.md to {out_dir}")


if __name__ == "__main__":
    main()
