"""Stage 4 (optional): Cohen's κ reliability check on a stratified sample.

Picks 6 predictions from each of the 5 most-populated Stage-2-ambiguous
categories (target 30 rows). Walks the user through a CLI review loop
and computes Cohen's κ against the LLM labels. Appends the result to
`report.md`.

Categories offered to the user: A1, A2, B1, and the two most frequent
rule-labeled categories in the current run (dynamically resolved).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.taxonomy import CATEGORY_CODES

DEFAULT_N_PER_CATEGORY = 6


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Inline Cohen's κ — no sklearn dependency.

    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement rate, p_e = expected agreement by chance.
    """
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    expected = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _interpret(kappa: float) -> str:
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def _sample(rows: list[dict], n_per: int, seed: int) -> list[dict]:
    """Pick up to `n_per` rows from each of 5 target categories."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("primary_cause")
        if c is None:
            continue
        by_cat.setdefault(c, []).append(r)

    # Target categories: the 3 Stage-2 ones + top 2 rule-labeled.
    rule_counts = Counter(
        r.get("primary_cause") for r in rows
        if r.get("primary_cause") not in {"A1", "A2", "B1", None}
    )
    top_rule = [c for c, _ in rule_counts.most_common(2)]
    targets = ["A1", "A2", "B1"] + top_rule

    rng = random.Random(seed)
    out: list[dict] = []
    for c in targets:
        pool = by_cat.get(c, [])
        if not pool:
            continue
        rng.shuffle(pool)
        out.extend(pool[:n_per])
    return out


def _display(row: dict) -> None:
    print("═" * 70)
    print(f"QID {row['question_id']}   condition={row['condition']} model={row['model']}")
    print(f"scope={row.get('scope')}  hop={row.get('hop_count')}  f1={row.get('f1')}")
    print(f"Q: {row.get('question')}")
    print(f"A (gold): {row.get('gold')}")
    print(f"Pred: {row.get('prediction')}")
    reason = row.get("reason")
    if reason:
        print(f"LLM reason: {reason}")
    print(f"LLM label: {row['primary_cause']}  (secondary: {row.get('secondary_cause')})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--out",
                        default="data/eval/failure_taxonomy/kappa_sample.jsonl")
    parser.add_argument("--report",
                        default="data/eval/failure_taxonomy/report.md")
    parser.add_argument("--n-per-category", type=int, default=DEFAULT_N_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.classified) if l.strip()]
    sample = _sample(rows, args.n_per_category, args.seed)
    if not sample:
        raise SystemExit("no rows available for sampling (empty classified file?)")

    labeled: list[dict] = []
    print(f"You will review {len(sample)} predictions. "
          f"For each, type one of: {CATEGORY_CODES} or 'skip'.\n")
    try:
        for i, row in enumerate(sample, 1):
            print(f"\n[{i}/{len(sample)}]")
            _display(row)
            while True:
                choice = input(f"Your label ({'|'.join(CATEGORY_CODES)} or skip): ").strip().upper()
                if choice == "SKIP":
                    break
                if choice in CATEGORY_CODES:
                    labeled.append({**row, "human_label": choice})
                    break
                print("  ? unknown code; try again.")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — saving what we have.")

    Path(args.out).write_text(
        "\n".join(json.dumps(r) for r in labeled) + "\n", encoding="utf-8"
    )
    if not labeled:
        raise SystemExit("no labels collected; κ not computed.")

    llm = [r["primary_cause"] for r in labeled]
    human = [r["human_label"] for r in labeled]
    k = cohen_kappa(llm, human)
    interp = _interpret(k)
    print(f"\nCohen's κ over {len(labeled)} rows: {k:.3f} ({interp})")

    # Rewrite the reliability section of report.md.
    report_path = Path(args.report)
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        block = (
            "## Reliability\n\n"
            f"Cohen's κ (LLM vs. human) on {len(labeled)} stratified samples: "
            f"**{k:.3f}** ({interp}).\n\n"
            f"Sample file: [kappa_sample.jsonl]({Path(args.out).name}).\n"
        )
        # Replace the placeholder if it's there; else append.
        if "## Reliability" in text:
            start = text.index("## Reliability")
            end = text.find("\n## ", start + 1)
            end = end if end != -1 else len(text)
            text = text[:start] + block + ("\n" + text[end:] if end != len(text) else "\n")
        else:
            text = text.rstrip() + "\n\n" + block
        report_path.write_text(text, encoding="utf-8")
        print(f"Report updated: {report_path}")


if __name__ == "__main__":
    main()
