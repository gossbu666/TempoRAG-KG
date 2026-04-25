"""Generate publication-ready PNG figures from the eval + taxonomy artefacts.

Outputs to `docs/figures/`:
  - fig_2x2_ablation.png      — 7 models × 4 conditions F1 heatmap
  - fig_condition_avg.png     — 7-model average F1 per condition (with CI)
  - fig_by_hop.png            — by-hop F1 lines (gpt-4.1-nano, 4 conditions)
  - fig_by_scope.png          — by-scope F1 grouped bars (gpt-4.1-nano)
  - fig_taxonomy_by_cond.png  — failure category counts × condition (stacked)
  - fig_taxonomy_by_model.png — failure category counts × model (stacked)

Re-runs are idempotent. Reads from:
  - data/eval/{vanilla,timefilter,kg2rag,temporag}/<model>/{summary.json,
    predictions.jsonl}
  - data/eval/failure_taxonomy/by_*.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "data" / "eval"
TAX_DIR = EVAL_ROOT / "failure_taxonomy"
FIG_DIR = REPO_ROOT / "docs" / "figures"

CONDITIONS = ["vanilla", "timefilter", "kg2rag", "temporag"]
COND_LABELS = {"vanilla": "L0 Vanilla", "timefilter": "L1 TimeFilter",
               "kg2rag": "L2 KG²RAG", "temporag": "L3 TempoRAG-KG"}
MODELS_ORDER = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o",
                "llama-70b", "llama-8b", "gpt-oss-120b", "gpt-oss-20b"]


def _summary(condition: str, model: str) -> dict:
    p = EVAL_ROOT / condition / model / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _preds(condition: str, model: str) -> list[dict]:
    p = EVAL_ROOT / condition / model / "predictions.jsonl"
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _qa_records() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in ("data/qa/home_grown.jsonl", "data/qa/multihop_filtered.jsonl"):
        p = REPO_ROOT / src
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[str(r.get("question_id"))] = r
    return out


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.relative_to(REPO_ROOT)}")


# ──────────────────────────────────────────────────────────────────────
# Figure 1: 7×4 F1 heatmap
# ──────────────────────────────────────────────────────────────────────

def fig_2x2_ablation() -> None:
    grid = np.zeros((len(MODELS_ORDER), len(CONDITIONS)))
    for i, m in enumerate(MODELS_ORDER):
        for j, c in enumerate(CONDITIONS):
            grid[i, j] = _summary(c, m)["f1_mean"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.10, vmax=0.27, aspect="auto")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=15, ha="right")
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels(MODELS_ORDER)
    for i in range(len(MODELS_ORDER)):
        for j in range(len(CONDITIONS)):
            v = grid[i, j]
            color = "black" if v > 0.18 else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Token-F1 (mean)", rotation=270, labelpad=14)
    ax.set_title("F1 across 7 models × 4 retrieval conditions (n=129 Qs)")
    plt.tight_layout()
    _save(fig, "fig_2x2_ablation.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: per-condition mean F1 with bootstrap CI (averaged across 7 models)
# ──────────────────────────────────────────────────────────────────────

def fig_condition_avg() -> None:
    means: dict[str, list[float]] = defaultdict(list)
    cis_lo: dict[str, list[float]] = defaultdict(list)
    cis_hi: dict[str, list[float]] = defaultdict(list)
    for c in CONDITIONS:
        for m in MODELS_ORDER:
            s = _summary(c, m)
            means[c].append(s["f1_mean"])
            cis_lo[c].append(s["f1_ci"][0])
            cis_hi[c].append(s["f1_ci"][1])

    avg = {c: np.mean(means[c]) for c in CONDITIONS}
    err_lo = {c: avg[c] - np.mean(cis_lo[c]) for c in CONDITIONS}
    err_hi = {c: np.mean(cis_hi[c]) - avg[c] for c in CONDITIONS}

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(CONDITIONS))
    vals = [avg[c] for c in CONDITIONS]
    yerr = [[err_lo[c] for c in CONDITIONS], [err_hi[c] for c in CONDITIONS]]
    colors = ["#7d7d7d", "#3a8a45", "#b85450", "#1f6cb6"]
    ax.bar(xs, vals, yerr=yerr, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Token-F1 (7-model avg ± 95% CI)")
    ax.set_title("Average F1 by retrieval condition\n"
                 "(L1 TimeFilter best; L2 KG²RAG regresses; L3 ≈ L1)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.18)
    plt.tight_layout()
    _save(fig, "fig_condition_avg.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: by-hop F1 (gpt-4.1-nano)
# ──────────────────────────────────────────────────────────────────────

def fig_by_hop() -> None:
    qa = _qa_records()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"vanilla": "#7d7d7d", "timefilter": "#3a8a45",
              "kg2rag": "#b85450", "temporag": "#1f6cb6"}
    markers = {"vanilla": "o", "timefilter": "s", "kg2rag": "^", "temporag": "D"}

    for c in CONDITIONS:
        rows = _preds(c, "gpt-4.1-nano")
        by_hop: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            qid = str(r.get("question_id"))
            hop = (r.get("hop_count") or qa.get(qid, {}).get("hop_count") or 1)
            by_hop[hop].append(float(r.get("f1") or 0.0))
        hops = sorted(by_hop)
        f1s = [np.mean(by_hop[h]) for h in hops]
        ax.plot(hops, f1s, label=COND_LABELS[c], color=colors[c],
                marker=markers[c], linewidth=2, markersize=9)
        for h, f in zip(hops, f1s):
            ax.annotate(f"{f:.2f}", (h, f), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=colors[c])

    ax.set_xlabel("Hop count")
    ax.set_ylabel("Token-F1 (mean)")
    ax.set_title("F1 by hop count — gpt-4.1-nano (n=129)")
    ax.set_xticks([1, 2, 3])
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_by_hop.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: by-scope grouped bars (gpt-4.1-nano)
# ──────────────────────────────────────────────────────────────────────

def fig_by_scope() -> None:
    scopes = ["intra", "inter_year", "cross_company",
              "fiscal_vs_calendar", "forward_looking"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#7d7d7d", "#3a8a45", "#b85450", "#1f6cb6"]
    bar_w = 0.20
    xs = np.arange(len(scopes))

    for i, c in enumerate(CONDITIONS):
        rows = _preds(c, "gpt-4.1-nano")
        f1_by_scope: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            f1_by_scope[r.get("scope", "?")].append(float(r.get("f1") or 0.0))
        vals = [np.mean(f1_by_scope[s]) if f1_by_scope[s] else 0.0 for s in scopes]
        ax.bar(xs + (i - 1.5) * bar_w, vals, bar_w,
               label=COND_LABELS[c], color=colors[i], edgecolor="black", linewidth=0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_", "\n") for s in scopes])
    ax.set_ylabel("Token-F1 (mean)")
    ax.set_title("F1 by scope — gpt-4.1-nano (n=129)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_by_scope.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 5: taxonomy stacked bars by condition
# ──────────────────────────────────────────────────────────────────────

CAT_COLORS = {
    "A1": "#d62728", "A2": "#ff7f0e", "A3": "#bcbd22",
    "A4": "#9467bd", "A5": "#7f7f7f",
    "B1": "#1f77b4", "B2": "#17becf", "B3": "#2ca02c",
    "B4": "#8c564b", "B5": "#e377c2", "NF": "#bbbbbb",
}
CAT_LABELS = {
    "A1": "A1 retrieval miss", "A2": "A2 hallucination",
    "A3": "A3 tersification", "A4": "A4 IDK-when-answerable",
    "A5": "A5 parse error",
    "B1": "B1 fact absent", "B2": "B2 forward-looking",
    "B3": "B3 fiscal/calendar", "B4": "B4 out-of-scope",
    "B5": "B5 cross-filing", "NF": "NF non-failure",
}


def _read_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    field = header[0]
    cats = header[1:-1]  # drop 'total'
    rows = reader[1:]
    keys = [r[0] for r in rows]
    grid = np.array([[int(v or 0) for v in r[1:-1]] for r in rows], dtype=int)
    return keys, cats, grid


def _stacked(field: str, key_label: str, csv_name: str, out_name: str,
             title_suffix: str) -> None:
    csv_path = TAX_DIR / csv_name
    if not csv_path.exists():
        print(f"  ! missing {csv_path}, skipping")
        return
    keys, cats, grid = _read_csv(csv_path)
    # Cats include 'unlabeled' too; drop empty columns.
    present_cats = [c for i, c in enumerate(cats) if grid[:, i].sum() > 0]
    grid = grid[:, [cats.index(c) for c in present_cats]]
    cats = present_cats

    if field == "by_condition":
        # Reorder L0/L1/L2/L3 explicitly.
        order = ["L0", "L1", "L2", "L3"]
        keys = [k for k in order if k in keys]
        grid = np.array([grid[order.index(k)] for k in keys])
    elif field == "by_model":
        keys = [k for k in MODELS_ORDER if k in keys]
        # remap rows
        with csv_path.open() as f:
            reader = list(csv.reader(f))
        original = [r[0] for r in reader[1:]]
        grid = np.array([grid[original.index(k)] for k in keys])

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(keys))
    xs = np.arange(len(keys))
    for c in cats:
        col_idx = present_cats.index(c)
        vals = grid[:, col_idx]
        color = CAT_COLORS.get(c, "#999999")
        ax.bar(xs, vals, bottom=bottoms, color=color, label=CAT_LABELS.get(c, c),
               edgecolor="white", linewidth=0.4)
        bottoms = bottoms + vals
    ax.set_xticks(xs)
    ax.set_xticklabels(keys)
    ax.set_xlabel(key_label)
    ax.set_ylabel("# predictions")
    ax.set_title(f"Failure category distribution {title_suffix}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_name)


def fig_taxonomy_by_cond() -> None:
    _stacked("by_condition", "Retrieval condition",
             "by_condition.csv", "fig_taxonomy_by_cond.png",
             "× condition (3,607 predictions)")


def fig_taxonomy_by_model() -> None:
    _stacked("by_model", "Model",
             "by_model.csv", "fig_taxonomy_by_model.png",
             "× model (4 conditions × 129 Qs)")


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating figures →", FIG_DIR)
    fig_2x2_ablation()
    fig_condition_avg()
    fig_by_hop()
    fig_by_scope()
    fig_taxonomy_by_cond()
    fig_taxonomy_by_model()
    print("Done.")


if __name__ == "__main__":
    main()
