"""Filter FinReflectKG-MultiHop (555 QA) to our 10-ticker subset.

Keeps only questions whose every hop's `source_file` refers to a 10-K filing
of one of our target tickers. The filter is strict (ALL hops in set) because
a partial match would leak a company we haven't extracted a KG for — the
generator would then be asked about facts our KG cannot possibly contain.

Output: `data/qa/multihop_filtered.jsonl`, one JSON per line with fields:

    question_id, question, answer, hop_count, scope, tickers, years,
    pattern, source_chunks, evidence

`scope` is a short label we reuse in `aggregate_by_scope` (mapping from the
paper's `document_relationship` field). `source_chunks` and `evidence` are
carried through so the eval pipeline can attribute model answers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = REPO_ROOT / "data" / "multihop_qa" / "final_master_dataset.json"
OUT_PATH = REPO_ROOT / "data" / "qa" / "multihop_filtered.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "multihop_filter_report.md"

TARGET_TICKERS = frozenset({
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "CSCO", "ORCL", "INTC", "NVDA", "ADBE",
})

TICKER_RE = re.compile(r"^([A-Z]+)_10k_(\d{4})\.pdf$")

# Map the paper's category label → the short scope label we use internally.
# `aggregate_by_scope` in src.eval keys on this.
SCOPE_MAP = {
    "intra_document": "intra",
    "inter_document_same_company": "inter_year",
    "inter_document_cross_company": "cross_company",
}

HOP_KEYS = ("hop_1_rel", "hop_2_rel", "hop_3_rel")


def tickers_and_years(q: dict) -> tuple[set[str], set[int]]:
    """Return (tickers_mentioned, years_mentioned) across all hops."""
    tickers: set[str] = set()
    years: set[int] = set()
    pd = q.get("path_data", {})
    for key in HOP_KEYS:
        hop = pd.get(key)
        if isinstance(hop, dict):
            m = TICKER_RE.match(hop.get("source_file", ""))
            if m:
                tickers.add(m.group(1))
                years.add(int(m.group(2)))
    return tickers, years


def flatten_record(q: dict) -> dict:
    """Shrink the master record to the eval-facing fields.

    We drop `entities`/`entity_types`/`path_data.{start_node,intermediate_node,
    end_node}` because the eval pipeline only needs question/answer/scope plus
    enough source pointers to show the judge (for reason auditing) and to build
    an oracle context baseline if we want one.
    """
    pd = q.get("path_data", {})
    source_chunks = []
    evidence = []
    for key in HOP_KEYS:
        hop = pd.get(key)
        if isinstance(hop, dict):
            source_chunks.append({
                "source_file": hop.get("source_file"),
                "page_id": hop.get("page_id"),
                "chunk_id": hop.get("chunk_id"),
            })
            txt = hop.get("chunk_text")
            if txt:
                evidence.append(txt)
    tickers, years = tickers_and_years(q)
    paper_cat = q.get("document_relationship", "")
    return {
        "question_id": q["question_id"],
        "question": q["question"],
        "answer": q["answer"],
        "hop_count": q.get("hop_count"),
        "scope": SCOPE_MAP.get(paper_cat, paper_cat or "unknown"),
        "paper_category": paper_cat,
        "tickers": sorted(tickers),
        "years": sorted(years),
        "pattern": q.get("pattern"),
        "source_chunks": source_chunks,
        "evidence": evidence,
        "source_dataset": "finreflectkg_multihop_555",
    }


def write_report(kept: list[dict], total: int, by_scope: Counter, by_ticker: Counter) -> None:
    lines = [
        "# FinReflectKG-MultiHop Filter Report",
        "",
        f"**Source:** `{IN_PATH.relative_to(REPO_ROOT)}` (555 QA pairs)",
        f"**Filter:** all hops' source filings in {sorted(TARGET_TICKERS)}",
        f"**Output:** `{OUT_PATH.relative_to(REPO_ROOT)}`",
        f"**Kept:** {len(kept):,} / {total:,}",
        "",
        "## By scope",
        "",
        "| Scope | Kept |",
        "|---|---:|",
    ]
    for scope in ("intra", "inter_year", "cross_company"):
        lines.append(f"| {scope} | {by_scope.get(scope, 0):,} |")
    lines += [
        "",
        "## By ticker (kept questions touching each ticker)",
        "",
        "| Ticker | Qs touching |",
        "|---|---:|",
    ]
    for ticker in sorted(TARGET_TICKERS):
        lines.append(f"| {ticker} | {by_ticker.get(ticker, 0):,} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Strict filter: every hop in a kept question references a 10-K whose "
        "ticker is in our target set. A question that mentions a ticker outside "
        "the set (even in one hop) is dropped, because our KG has no entries "
        "for that company and the generator would be asked to answer from facts "
        "we don't carry.",
        "- `scope` field: re-mapped from the paper's `document_relationship`. "
        "Used as the stratification key by `src.eval.aggregate_by_scope`.",
        "- The `cross_company` bucket was 0 under the Mag5 filter; the 10-ticker "
        "expansion (adding CSCO/ORCL/INTC/NVDA/ADBE) unlocks this scope and "
        "lets us measure the temporal-KG lift on its most-distinctive regime.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file.",
    )
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"{OUT_PATH} already exists — pass --force to regenerate.")
        return

    with IN_PATH.open("r", encoding="utf-8") as f:
        master = json.load(f)
    questions = master["questions"]
    total = len(questions)

    kept: list[dict] = []
    by_scope: Counter = Counter()
    by_ticker: Counter = Counter()
    for q in questions:
        tickers, _years = tickers_and_years(q)
        if tickers and tickers <= TARGET_TICKERS:
            rec = flatten_record(q)
            kept.append(rec)
            by_scope[rec["scope"]] += 1
            for t in tickers:
                by_ticker[t] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_report(kept, total, by_scope, by_ticker)

    print(f"Kept {len(kept):,} / {total:,} questions")
    print(f"By scope: {dict(by_scope)}")
    print(f"By ticker: {dict(by_ticker)}")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
