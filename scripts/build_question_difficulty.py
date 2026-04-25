"""Per-question difficulty analysis: rank questions by average F1
across all (model × condition) cells, then surface easy/hard patterns.

Outputs `docs/question_difficulty_analysis.md`:
  1. Top-10 EASIEST questions (avg F1 ≈ 1.0).
  2. Top-10 HARDEST questions (avg F1 ≈ 0.0).
  3. Per-bucket aggregates: scope distribution + hop distribution +
     ticker distribution in each top-10.
  4. Concrete examples (question + gold) for the head and tail.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "data" / "eval"
QA_PATHS = [
    REPO_ROOT / "data" / "qa" / "home_grown.jsonl",
    REPO_ROOT / "data" / "qa" / "multihop_filtered.jsonl",
]
OUT = REPO_ROOT / "docs" / "question_difficulty_analysis.md"

CONDITIONS = ["vanilla", "timefilter", "kg2rag", "temporag"]
COND_LABEL = {"vanilla": "L0", "timefilter": "L1",
              "kg2rag": "L2", "temporag": "L3"}
MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o",
          "llama-70b", "llama-8b", "gpt-oss-120b", "gpt-oss-20b"]


def _qa_records() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in QA_PATHS:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[str(r["question_id"])] = r
    return out


def _gather_f1s() -> dict[str, dict]:
    """{qid: {"f1_per_cell": [...], "per_condition": {L0:[...], L1:..., ...}}}"""
    out: dict[str, dict] = defaultdict(
        lambda: {"f1_per_cell": [], "per_condition": defaultdict(list)}
    )
    for cond in CONDITIONS:
        cond_label = COND_LABEL[cond]
        for model in MODELS:
            preds_path = EVAL_ROOT / cond / model / "predictions.jsonl"
            if not preds_path.exists():
                continue
            with preds_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    qid = str(r["question_id"])
                    f1 = float(r.get("f1") or 0.0)
                    out[qid]["f1_per_cell"].append(f1)
                    out[qid]["per_condition"][cond_label].append(f1)
    return out


def _summarize_bucket(qids: list[str], qa: dict[str, dict]) -> dict:
    """Aggregate scope / hop / ticker distribution."""
    scopes = Counter()
    hops = Counter()
    tickers = Counter()
    for qid in qids:
        rec = qa.get(qid, {})
        scopes[rec.get("scope") or "?"] += 1
        hops[rec.get("hop_count") or rec.get("num_hops") or "?"] += 1
        for t in rec.get("tickers") or []:
            tickers[t] += 1
    return {"scopes": scopes, "hops": hops, "tickers": tickers}


def _format_example_block(qid: str, rec: dict, stats: dict) -> str:
    avg = statistics.mean(stats["f1_per_cell"]) if stats["f1_per_cell"] else 0.0
    per_cond = {c: (statistics.mean(v) if v else 0.0)
                for c, v in stats["per_condition"].items()}
    gold = rec.get("answer")
    if isinstance(gold, list):
        gold = " | ".join(str(g) for g in gold)
    gold = str(gold or "")[:200]

    cond_str = "  ".join(f"{c}={per_cond.get(c, 0):.2f}"
                         for c in ("L0", "L1", "L2", "L3"))
    scope = rec.get("scope") or "?"
    hop = rec.get("hop_count") or rec.get("num_hops") or "?"
    return (
        f"### qid {qid} — avg F1 {avg:.3f}  ({cond_str})\n"
        f"- **Scope / hop:** {scope} / {hop}\n"
        f"- **Q:** {rec.get('question', '?')}\n"
        f"- **Gold:** {gold}\n"
    )


def main() -> None:
    qa = _qa_records()
    gathered = _gather_f1s()
    if not gathered:
        raise SystemExit("no predictions found.")

    qid_avg = []
    for qid, stats in gathered.items():
        cells = stats["f1_per_cell"]
        if not cells:
            continue
        qid_avg.append((qid, statistics.mean(cells), stats))
    qid_avg.sort(key=lambda x: x[1])

    n_total = len(qid_avg)
    hardest = qid_avg[:10]
    easiest = qid_avg[-10:][::-1]

    hard_bucket = _summarize_bucket([q for q, _, _ in hardest], qa)
    easy_bucket = _summarize_bucket([q for q, _, _ in easiest], qa)

    lines = [
        "# Per-question difficulty analysis",
        "",
        "Each question's difficulty score = mean Token-F1 across all "
        f"{len(MODELS)} models $\\times$ {len(CONDITIONS)} retrieval conditions = "
        f"{len(MODELS)*len(CONDITIONS)} prediction cells. Lower = harder.",
        "",
        f"Total questions analysed: **{n_total}**.",
        "",
        "## 1. Top-10 hardest questions (lowest mean F1)",
        "",
        "| Rank | QID | Avg F1 | Scope | Hop | Ticker(s) |",
        "|---:|---|---:|---|---:|---|",
    ]
    for i, (qid, avg, _) in enumerate(hardest, 1):
        rec = qa.get(qid, {})
        tickers = ",".join(rec.get("tickers") or [])
        lines.append(
            f"| {i} | {qid} | {avg:.3f} | {rec.get('scope', '?')} | "
            f"{rec.get('hop_count') or rec.get('num_hops') or '?'} | {tickers} |"
        )

    lines += ["", "### Pattern in hardest 10",
              f"- **Scope distribution:** {dict(hard_bucket['scopes'])}",
              f"- **Hop distribution:** {dict(hard_bucket['hops'])}",
              f"- **Top tickers:** {dict(hard_bucket['tickers'].most_common(5))}",
              ""]

    lines += [
        "## 2. Top-10 easiest questions (highest mean F1)",
        "",
        "| Rank | QID | Avg F1 | Scope | Hop | Ticker(s) |",
        "|---:|---|---:|---|---:|---|",
    ]
    for i, (qid, avg, _) in enumerate(easiest, 1):
        rec = qa.get(qid, {})
        tickers = ",".join(rec.get("tickers") or [])
        lines.append(
            f"| {i} | {qid} | {avg:.3f} | {rec.get('scope', '?')} | "
            f"{rec.get('hop_count') or rec.get('num_hops') or '?'} | {tickers} |"
        )

    lines += ["", "### Pattern in easiest 10",
              f"- **Scope distribution:** {dict(easy_bucket['scopes'])}",
              f"- **Hop distribution:** {dict(easy_bucket['hops'])}",
              f"- **Top tickers:** {dict(easy_bucket['tickers'].most_common(5))}",
              ""]

    # Concrete examples
    lines.append("## 3. Worked examples")
    lines.append("")
    lines.append("### Three hardest")
    lines.append("")
    for qid, _, stats in hardest[:3]:
        lines.append(_format_example_block(qid, qa.get(qid, {}), stats))
    lines.append("### Three easiest")
    lines.append("")
    for qid, _, stats in easiest[:3]:
        lines.append(_format_example_block(qid, qa.get(qid, {}), stats))

    # Aggregate distribution
    lines += [
        "## 4. Difficulty distribution",
        "",
        f"- Mean F1 across all {n_total} questions: "
        f"**{statistics.mean(a for _, a, _ in qid_avg):.3f}**",
        f"- Median F1: **{statistics.median(a for _, a, _ in qid_avg):.3f}**",
        f"- Std dev:   **{statistics.stdev(a for _, a, _ in qid_avg):.3f}**",
        f"- Questions with avg F1 == 0 (every cell failed): "
        f"**{sum(1 for _, a, _ in qid_avg if a == 0)}**",
        f"- Questions with avg F1 \\geq 0.8 (consistently solved): "
        f"**{sum(1 for _, a, _ in qid_avg if a >= 0.8)}**",
        "",
    ]

    # Save also a per-qid JSONL for downstream tools.
    perqid_path = REPO_ROOT / "data" / "eval" / "question_difficulty.jsonl"
    perqid_path.parent.mkdir(parents=True, exist_ok=True)
    with perqid_path.open("w", encoding="utf-8") as f:
        for qid, avg, stats in qid_avg:
            rec = qa.get(qid, {})
            per_cond = {c: (statistics.mean(v) if v else 0.0)
                        for c, v in stats["per_condition"].items()}
            f.write(json.dumps({
                "question_id": qid,
                "avg_f1": avg,
                "per_condition_f1": per_cond,
                "scope": rec.get("scope"),
                "hop_count": rec.get("hop_count") or rec.get("num_hops"),
                "tickers": rec.get("tickers"),
                "years": rec.get("years"),
                "n_cells": len(stats["f1_per_cell"]),
            }, ensure_ascii=False) + "\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT.relative_to(REPO_ROOT)}")
    print(f"Per-qid jsonl: {perqid_path.relative_to(REPO_ROOT)}")
    print()
    print("Summary:")
    print(f"  Mean F1: {statistics.mean(a for _, a, _ in qid_avg):.3f}")
    print(f"  F1==0:   {sum(1 for _, a, _ in qid_avg if a == 0)}")
    print(f"  F1>=0.8: {sum(1 for _, a, _ in qid_avg if a >= 0.8)}")


if __name__ == "__main__":
    main()
