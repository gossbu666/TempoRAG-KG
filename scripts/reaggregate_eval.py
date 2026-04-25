"""Re-aggregate predictions.jsonl → summary.json + report.md with new metrics.

Runs after `src/eval.py` is extended (e.g. adding `coverage`, `f1_answered`).
No API calls — reads per-row `prediction` + `gold` from existing
`predictions.jsonl` files and recomputes aggregate stats.

Usage:
    python -m scripts.reaggregate_eval --dir data/eval/vanilla
    python -m scripts.reaggregate_eval --dir data/eval/timefilter
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval import aggregate

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_preds(path: Path) -> tuple[list[str], list]:
    preds: list[str] = []
    golds: list = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds.append(rec["prediction"])
            golds.append(rec["gold"])
    return preds, golds


def _reagg_one(model_dir: Path) -> dict | None:
    preds_path = model_dir / "predictions.jsonl"
    if not preds_path.exists():
        return None
    preds, golds = _load_preds(preds_path)
    if not preds:
        return None
    summary = aggregate(preds, golds)
    summary["model"] = model_dir.name

    # Preserve provider / model_id / cache_hits / elapsed_sec if present in
    # the prior summary — these come from the orchestrator, not eval.
    old_path = model_dir / "summary.json"
    if old_path.exists():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8"))
            for k in ("model_id", "provider", "cache_hits", "elapsed_sec"):
                if k in old and k not in summary:
                    summary[k] = old[k]
        except json.JSONDecodeError:
            pass

    old_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _write_report(summaries: list[dict], out_dir: Path, *, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "**Coverage** = fraction of rows where the model attempted an answer "
        "(non-IDK, non-empty). **F1@answered** = token-F1 on the answered "
        "subset — factors out retrieval-coverage failures from answer-quality "
        "failures.",
        "",
        "| Model | Provider | F1 | F1 CI | Coverage | F1@answered | n_answered |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda s: -s["f1_mean"]):
        f1_lo, f1_hi = s["f1_ci"]
        fa_lo, fa_hi = s.get("f1_answered_ci", (0.0, 0.0))
        lines.append(
            f"| `{s['model']}` | {s.get('provider','?')} | "
            f"{s['f1_mean']:.3f} | [{f1_lo:.3f}, {f1_hi:.3f}] | "
            f"{s.get('coverage', 0.0):.1%} | "
            f"{s.get('f1_answered_mean', 0.0):.3f} [{fa_lo:.3f}, {fa_hi:.3f}] | "
            f"{s.get('n_answered', 0)} |"
        )
    lines += ["", f"_Re-aggregated: {time.strftime('%Y-%m-%d %H:%M:%S')}_"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", required=True,
        help="Eval output directory containing per-model subdirs.",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        raise SystemExit(f"not found: {root}")

    summaries: list[dict] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        s = _reagg_one(model_dir)
        if s is None:
            print(f"  skip {model_dir.name} (no predictions.jsonl)")
            continue
        summaries.append(s)
        print(
            f"  {model_dir.name:14s}  F1={s['f1_mean']:.3f}  "
            f"cov={s.get('coverage', 0):.1%}  "
            f"F1@ans={s.get('f1_answered_mean', 0):.3f}  "
            f"n_ans={s.get('n_answered', 0)}"
        )

    title = args.title or f"{root.name.title()} RAG Evaluation"
    _write_report(summaries, root, title=title)
    print(f"\nReport: {(root / 'report.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
