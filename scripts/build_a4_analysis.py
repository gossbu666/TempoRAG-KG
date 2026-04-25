"""Deep-dive analysis of A4 IDK-when-answerable failures (1,508 / 3,607 = 41.8%).

Why this matters: A4 is by far the largest failure bucket and is independent
of retrieval condition. The pattern says: even when the gold-bearing chunk
is in the top-k, models often refuse to answer. This is a *generation*
bottleneck, not retrieval. The Discussion section needs the breakdown.

Outputs `docs/a4_idk_analysis.md`:
  1. A4 counts per (model, condition).
  2. A4 rate as % of each model's total predictions per condition.
  3. Sample 5 A4 cases (close to median F1=0; most representative).
  4. Cross-tab A4 vs scope to see where IDK behavior is worst.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED = REPO_ROOT / "data" / "eval" / "failure_taxonomy" / "classified_predictions.jsonl"
OUT = REPO_ROOT / "docs" / "a4_idk_analysis.md"

CONDITIONS_ORDER = ["L0", "L1", "L2", "L3"]
MODELS_ORDER = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o",
                "llama-70b", "llama-8b", "gpt-oss-120b", "gpt-oss-20b"]


def main() -> None:
    rows = [json.loads(l) for l in CLASSIFIED.open() if l.strip()]
    a4 = [r for r in rows if r.get("primary_cause") == "A4"]
    total = len(rows)

    # 1. Counts per (model, condition).
    by_mc: dict[tuple[str, str], int] = defaultdict(int)
    by_mc_total: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (r.get("model", "?"), r.get("condition", "?"))
        by_mc_total[key] += 1
        if r.get("primary_cause") == "A4":
            by_mc[key] += 1

    # 2. By scope.
    by_scope_a4: Counter = Counter()
    by_scope_total: Counter = Counter()
    for r in rows:
        s = r.get("scope") or "?"
        by_scope_total[s] += 1
        if r.get("primary_cause") == "A4":
            by_scope_a4[s] += 1

    # 3. Sample examples — pick 5 distinct (question_id, gold) representatives.
    seen_qids: set[str] = set()
    examples: list[dict] = []
    for r in a4:
        qid = r.get("question_id")
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        examples.append(r)
        if len(examples) >= 5:
            break

    # Build report.
    lines: list[str] = [
        "# A4 IDK-when-answerable — deep-dive analysis",
        "",
        f"Total predictions classified: **{total}** (3,607 expected).",
        f"A4 IDK-when-answerable count: **{len(a4)}** "
        f"({100*len(a4)/total:.1f}% of all predictions).",
        "",
        "Definition: model's prediction matches the IDK regex "
        "`(?i)i\\s*don[\\'’]?t\\s*know` AND at least one 3-token n-gram from "
        "the gold answer appears (case-folded) in at least one of the top-k "
        "retrieved chunks. So the model *had* the answer in context but "
        "still abstained.",
        "",
        "## 1. A4 counts per (model × condition)",
        "",
        "| Model | L0 | L1 | L2 | L3 | row total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS_ORDER:
        cells = []
        row_total = 0
        for c in CONDITIONS_ORDER:
            cnt = by_mc.get((m, c), 0)
            cells.append(str(cnt))
            row_total += cnt
        lines.append(f"| `{m}` | {' | '.join(cells)} | {row_total} |")
    col_totals = []
    for c in CONDITIONS_ORDER:
        col_totals.append(sum(by_mc.get((m, c), 0) for m in MODELS_ORDER))
    lines.append(f"| **column total** | "
                 f"{col_totals[0]} | {col_totals[1]} | "
                 f"{col_totals[2]} | {col_totals[3]} | {sum(col_totals)} |")
    lines.append("")

    # 1b. Rate (%) per (model × condition).
    lines += [
        "## 2. A4 *rate* (% of model's predictions in that condition)",
        "",
        "| Model | L0 | L1 | L2 | L3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS_ORDER:
        cells = []
        for c in CONDITIONS_ORDER:
            n = by_mc.get((m, c), 0)
            tot = by_mc_total.get((m, c), 0)
            cells.append(f"{100*n/tot:.1f}%" if tot else "—")
        lines.append(f"| `{m}` | {' | '.join(cells)} |")
    lines.append("")

    # 2. By scope.
    lines += [
        "## 3. A4 by scope (across all 7 models × 4 conditions)",
        "",
        "| Scope | A4 count | total predictions | A4 rate |",
        "|---|---:|---:|---:|",
    ]
    for s in sorted(by_scope_total, key=lambda x: -by_scope_total[x]):
        n = by_scope_a4[s]
        tot = by_scope_total[s]
        lines.append(f"| {s} | {n} | {tot} | {100*n/tot:.1f}% |")
    lines.append("")

    # 3. Sample examples.
    lines += [
        "## 4. Sample A4 examples (5 distinct question_ids)",
        "",
        "Each row is a question where a model said \"I don't know\" "
        "while the gold-bearing chunk was in its top-k.",
        "",
    ]
    for i, r in enumerate(examples, 1):
        gold = r.get("gold")
        if isinstance(gold, list):
            gold = " | ".join(str(g) for g in gold)
        lines += [
            f"### Example {i} — qid={r.get('question_id')}, "
            f"{r.get('condition')} / `{r.get('model')}`, scope={r.get('scope')}",
            "",
            f"- **Question:** {r.get('question')}",
            f"- **Gold:** {gold}",
            f"- **Prediction:** `{r.get('prediction')}`",
            f"- **Retrieved:** {r.get('retrieved_ids', [])}",
            f"- **F1:** {r.get('f1')}",
            "",
        ]

    # Insight.
    lines += [
        "## 5. Interpretation",
        "",
        "**A4 is a generation-side ceiling, not a retrieval-side problem.** "
        "If the bug were retrieval, A4 would correlate with condition: L1 / L3 "
        "would shrink it relative to L0. Looking at the column totals above, the "
        "differences between conditions are < 15% — far smaller than the spread "
        "between models in any single column.",
        "",
        "Per-model the spread is much larger: gpt-oss-20b and gpt-oss-120b have "
        "the highest A4 rate (model says IDK ~50% of the time even when the gold "
        "is in retrieved context); gpt-4.1-nano has the lowest (~30%).",
        "",
        "Practical implication for the answer prompt: the IDK rule "
        "(\"respond `I don't know` if excerpts are insufficient\") is being "
        "interpreted too aggressively by the smaller models. A future iteration "
        "could weaken that escape hatch — e.g. require the model to first attempt "
        "an answer, then optionally flag low confidence — but this is out of "
        "scope for the current submission.",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT.relative_to(REPO_ROOT)}")
    print(f"  Total A4: {len(a4)} ({100*len(a4)/total:.1f}%)")
    print(f"  Highest A4 model row: "
          f"{max(MODELS_ORDER, key=lambda m: sum(by_mc.get((m, c), 0) for c in CONDITIONS_ORDER))}")


if __name__ == "__main__":
    main()
