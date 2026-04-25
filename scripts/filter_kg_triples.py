"""T1: Filter pass on raw KG triples.

Drop noisy/unusable triples produced by the full extraction run so that
L2 (KG²RAG) and L3 (TempoRAG-KG) retrievers operate on a clean graph.

Drop rules (applied in order):
  1. Empty subject / predicate / object (after strip).
  2. Subject or object too short (<2 chars) — usually extraction noise.
  3. Object is a boolean literal ("true" / "false") — attribute placeholder,
     not a real entity/value.
  4. Object is a yes/no literal — same reason.
  5. Self-loop (subject == object, case-insensitive).
  6. Object longer than 300 chars — verbatim-evidence smuggled into object.

Input : data/kg/full/triples.jsonl
Output: data/kg/filtered/triples.jsonl + filter_report.md
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "kg" / "full" / "triples.jsonl"
OUT_DIR = REPO_ROOT / "data" / "kg" / "filtered"
OUT_TRIPLES = OUT_DIR / "triples.jsonl"
REPORT = OUT_DIR / "filter_report.md"

MIN_FIELD_LEN = 2
MAX_OBJECT_LEN = 300
BOOL_LITERALS = {"true", "false"}
YESNO_LITERALS = {"yes", "no"}


def should_drop(t: dict) -> str | None:
    s = str(t.get("subject", "")).strip()
    p = str(t.get("predicate", "")).strip()
    o = str(t.get("object", "")).strip()
    if not s or not p or not o:
        return "empty_field"
    if len(s) < MIN_FIELD_LEN or len(o) < MIN_FIELD_LEN:
        return "tooshort_sp"
    ol = o.lower()
    if ol in BOOL_LITERALS:
        return "bool_literal_object"
    if ol in YESNO_LITERALS:
        return "yesno_literal_object"
    if s.lower() == ol:
        return "self_loop"
    if len(o) > MAX_OBJECT_LEN:
        return "long_object"
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = Counter()
    per_ticker_kept = Counter()
    per_item_kept = Counter()
    per_fy_kept = Counter()
    temporal_kept = Counter()
    total = 0

    with SRC.open("r", encoding="utf-8") as fin, OUT_TRIPLES.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            t = json.loads(line)
            reason = should_drop(t)
            if reason is not None:
                dropped[reason] += 1
                continue
            fout.write(json.dumps(t, ensure_ascii=False) + "\n")
            kept += 1
            per_ticker_kept[t.get("ticker", "?")] += 1
            per_item_kept[t.get("item", "?")] += 1
            per_fy_kept[t.get("fy", "?")] += 1
            temporal_kept[t.get("metadata", {}).get("temporal_type", "?")] += 1

    total_dropped = sum(dropped.values())
    lines = [
        "# KG Triple Filter Report",
        "",
        f"**Input:** `{SRC.relative_to(REPO_ROOT)}`",
        f"**Output:** `{OUT_TRIPLES.relative_to(REPO_ROOT)}`",
        "",
        "## Headline",
        "",
        f"- Total input triples: **{total}**",
        f"- Kept: **{kept}** ({100*kept/total:.1f}%)",
        f"- Dropped: **{total_dropped}** ({100*total_dropped/total:.1f}%)",
        "",
        "## Drop reasons",
        "",
        "| Reason | Count | % of input |",
        "|---|---:|---:|",
    ]
    for reason, n in dropped.most_common():
        lines.append(f"| {reason} | {n} | {100*n/total:.2f}% |")

    lines += ["", "## Kept triples — by ticker", "",
              "| Ticker | Kept |", "|---|---:|"]
    for tk, n in sorted(per_ticker_kept.items()):
        lines.append(f"| {tk} | {n} |")

    lines += ["", "## Kept triples — by item", "",
              "| Item | Kept |", "|---|---:|"]
    for it, n in sorted(per_item_kept.items()):
        lines.append(f"| {it} | {n} |")

    lines += ["", "## Kept triples — by fiscal year", "",
              "| FY | Kept |", "|---|---:|"]
    for fy, n in sorted(per_fy_kept.items()):
        lines.append(f"| {fy} | {n} |")

    lines += ["", "## Kept triples — by temporal type", "",
              "| Temporal type | Kept | % of kept |", "|---|---:|---:|"]
    for tt, n in temporal_kept.most_common():
        lines.append(f"| {tt} | {n} | {100*n/kept:.1f}% |")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Total input: {total}")
    print(f"Kept: {kept} ({100*kept/total:.1f}%)")
    print(f"Dropped: {total_dropped} ({100*total_dropped/total:.1f}%)")
    for r, n in dropped.most_common():
        print(f"  {r}: {n}")
    print(f"\nOutput: {OUT_TRIPLES.relative_to(REPO_ROOT)}")
    print(f"Report: {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
