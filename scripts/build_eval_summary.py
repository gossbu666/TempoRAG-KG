"""Build a single comprehensive markdown summarizing all eval results.

Outputs `docs/eval_summary_2026_04_25.md` — feeds Report §5 directly.

Includes:
  1. Per-model overall F1 across 4 conditions (with bootstrap CIs)
  2. gpt-4.1-nano by-hop F1 breakdown × 4 conditions
  3. gpt-4.1-nano by-scope F1 breakdown × 4 conditions
  4. Failure-taxonomy headline counts × condition
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "data" / "eval"
TAX_DIR = EVAL_ROOT / "failure_taxonomy"
OUT = REPO_ROOT / "docs" / "eval_summary_2026_04_25.md"

CONDITIONS = ["vanilla", "timefilter", "kg2rag", "temporag"]
COND_LABELS = {"vanilla": "L0 Vanilla", "timefilter": "L1 TimeFilter",
               "kg2rag": "L2 KG²RAG", "temporag": "L3 TempoRAG-KG"}
MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o",
          "llama-70b", "llama-8b", "gpt-oss-120b", "gpt-oss-20b"]


def _summary(c: str, m: str) -> dict:
    return json.loads((EVAL_ROOT / c / m / "summary.json").read_text(encoding="utf-8"))


def _preds(c: str, m: str) -> list[dict]:
    out: list[dict] = []
    p = EVAL_ROOT / c / m / "predictions.jsonl"
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _qa() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in ("data/qa/home_grown.jsonl", "data/qa/multihop_filtered.jsonl"):
        p = REPO_ROOT / src
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[str(r.get("question_id"))] = r
    return out


def _per_model_overall() -> str:
    rows = ["## 1. Per-model overall F1 across 4 conditions",
            "",
            "All values are Token-F1 mean over n=129 questions, with 95% non-parametric "
            "bootstrap CI (n_resamples=1,000, fixed seed). Δ vs L0 in parentheses.",
            ""]
    rows.append("| Model | L0 Vanilla | L1 TimeFilter | L2 KG²RAG | L3 TempoRAG-KG |")
    rows.append("|---|---|---|---|---|")
    for m in MODELS:
        line = [f"`{m}`"]
        l0 = _summary("vanilla", m)["f1_mean"]
        for c in CONDITIONS:
            s = _summary(c, m)
            f1 = s["f1_mean"]
            lo, hi = s["f1_ci"]
            if c == "vanilla":
                line.append(f"{f1:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                delta_pct = 100 * (f1 - l0) / l0 if l0 > 0 else 0
                arrow = "↑" if delta_pct > 0 else ("↓" if delta_pct < 0 else "·")
                line.append(f"{f1:.3f} [{lo:.3f}, {hi:.3f}] ({arrow}{delta_pct:+.1f}%)")
        rows.append("| " + " | ".join(line) + " |")
    rows.append("")
    # 7-model average row
    rows.append("**7-model average F1** per condition:")
    rows.append("")
    rows.append("| Condition | Avg F1 | Δ vs L0 |")
    rows.append("|---|---|---|")
    avg_l0 = np.mean([_summary("vanilla", m)["f1_mean"] for m in MODELS])
    for c in CONDITIONS:
        avg = np.mean([_summary(c, m)["f1_mean"] for m in MODELS])
        delta = 100 * (avg - avg_l0) / avg_l0 if avg_l0 > 0 else 0
        rows.append(f"| {COND_LABELS[c]} | {avg:.3f} | {delta:+.1f}% |")
    return "\n".join(rows) + "\n"


def _by_hop_for_model(model: str) -> str:
    qa = _qa()
    rows = [f"## 2. By-hop F1 — {model}",
            "",
            "Hop count is taken from the QA record (or `hop_count` field on the "
            "prediction row when present). Each cell = mean F1 over the rows in that "
            "hop bucket.",
            ""]
    rows.append("| Hop | n | L0 | L1 | L2 | L3 |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    by_cond_hop: dict[str, dict[int, list[float]]] = {c: defaultdict(list) for c in CONDITIONS}
    n_per_hop: dict[int, int] = defaultdict(int)
    for c in CONDITIONS:
        preds = _preds(c, model)
        for r in preds:
            qid = str(r.get("question_id"))
            hop = r.get("hop_count") or qa.get(qid, {}).get("hop_count")
            if hop is None:
                continue
            by_cond_hop[c][hop].append(float(r.get("f1") or 0.0))
    # Use vanilla counts as canonical N per hop (others should match).
    for h, vals in by_cond_hop["vanilla"].items():
        n_per_hop[h] = len(vals)
    for h in sorted(n_per_hop):
        line = [str(h), str(n_per_hop[h])]
        for c in CONDITIONS:
            vals = by_cond_hop[c].get(h, [])
            line.append(f"{np.mean(vals):.3f}" if vals else "—")
        rows.append("| " + " | ".join(line) + " |")
    return "\n".join(rows) + "\n"


def _by_scope_for_model(model: str) -> str:
    rows = [f"## 3. By-scope F1 — {model}",
            "",
            "Scope is the labeled bucket from the QA record (intra / inter_year / "
            "cross_company / fiscal_vs_calendar / forward_looking).",
            ""]
    rows.append("| Scope | n | L0 | L1 | L2 | L3 |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    by_cond_scope: dict[str, dict[str, list[float]]] = {c: defaultdict(list) for c in CONDITIONS}
    n_per_scope: dict[str, int] = defaultdict(int)
    for c in CONDITIONS:
        preds = _preds(c, model)
        for r in preds:
            scope = r.get("scope") or "?"
            by_cond_scope[c][scope].append(float(r.get("f1") or 0.0))
    for s, vals in by_cond_scope["vanilla"].items():
        n_per_scope[s] = len(vals)
    for s in sorted(n_per_scope, key=lambda x: -n_per_scope[x]):
        line = [s, str(n_per_scope[s])]
        for c in CONDITIONS:
            vals = by_cond_scope[c].get(s, [])
            line.append(f"{np.mean(vals):.3f}" if vals else "—")
        rows.append("| " + " | ".join(line) + " |")
    return "\n".join(rows) + "\n"


def _taxonomy() -> str:
    csv_path = TAX_DIR / "by_condition.csv"
    if not csv_path.exists():
        return "## 4. Failure taxonomy\n\n_Not yet generated._\n"
    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    cats = header[1:-1]
    body = reader[1:]
    rows = ["## 4. Failure taxonomy — counts × condition",
            "",
            "Stage 1+2 of the failure-taxonomy classifier "
            "(`scripts/classify_failures_*.py`). 3,607 predictions over 7 models × 4 conditions × 129 Qs.",
            "",
            "| condition | " + " | ".join(cats) + " | total |",
            "|---|" + "---|" * (len(cats) + 1)]
    for row in body:
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parts = [
        f"# Final eval summary — TempoRAG-KG (snapshot 2026-04-25)\n",
        "Auto-generated by `scripts/build_eval_summary.py`.\n",
        "Sources:\n"
        "- `data/eval/{vanilla,timefilter,kg2rag,temporag}/<model>/{summary.json,predictions.jsonl}`\n"
        "- `data/eval/failure_taxonomy/by_condition.csv`\n",
        _per_model_overall(),
        _by_hop_for_model("gpt-4.1-nano"),
        _by_scope_for_model("gpt-4.1-nano"),
        _taxonomy(),
        "## Figures available\n"
        "- `docs/figures/fig_2x2_ablation.png`\n"
        "- `docs/figures/fig_condition_avg.png`\n"
        "- `docs/figures/fig_by_hop.png`\n"
        "- `docs/figures/fig_by_scope.png`\n"
        "- `docs/figures/fig_taxonomy_by_cond.png`\n"
        "- `docs/figures/fig_taxonomy_by_model.png`\n",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote: {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
