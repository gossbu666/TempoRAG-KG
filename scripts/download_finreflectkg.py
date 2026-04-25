"""Download the FinReflectKG subset we need as the baseline KG arm.

Fetches `domyn/FinReflectKG` from the HuggingFace Hub, streams it so we do
not materialize all 17.5M triples, and keeps only the 5 tickers x 3 years
(2022, 2023, 2024) we evaluate on. Writes one JSONL line per triple to
`data/finreflectkg/triples.jsonl` and a summary to
`docs/finreflectkg_subset_report.md`.

Why streaming:
    17.5M * ~1KB per record ≈ 17GB; we only want ~0.2% of it. Loading the
    full dataset into memory or disk before filtering is wasteful and slow
    on the first run. Streaming also lets the tqdm progress bar give real
    feedback even when the filter is very selective.

Re-running the script is safe: the output file is overwritten, not appended.
The HuggingFace dataset is immutable for a fixed snapshot, so re-running
from the same machine is byte-equivalent.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "domyn/FinReflectKG"
# 10-ticker tech-megacap scope (v2.2 expansion, 2026-04-19).
# Mag5 + CSCO/ORCL/INTC/NVDA/ADBE. Chosen to unlock the cross-company scope in
# the FinReflectKG-MultiHop filter — pairs like NVDA+ADBE close the bucket.
TARGET_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "CSCO", "ORCL", "INTC", "NVDA", "ADBE",
}
TARGET_YEARS = {2022, 2023, 2024}

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "finreflectkg"
OUT_PATH = OUT_DIR / "triples.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "finreflectkg_subset_report.md"


def matches_filter(row: dict) -> bool:
    """Row passes filter iff ticker is one of our 5 and year is 2022-2024."""
    return row.get("ticker") in TARGET_TICKERS and row.get("year") in TARGET_YEARS


def stream_filtered(dataset_id: str) -> Iterator[dict]:
    """Yield rows that match our ticker+year filter."""
    # `streaming=True` returns an IterableDataset that fetches shards on demand.
    ds = load_dataset(dataset_id, split="train", streaming=True)
    for row in ds:
        if matches_filter(row):
            yield row


def write_jsonl(rows: Iterator[dict], out_path: Path) -> tuple[int, Counter, Counter]:
    """Write matching rows to JSONL, returning (count, by_ticker, by_extraction_type)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_ticker: Counter = Counter()
    by_extraction_type: Counter = Counter()
    by_ticker_year: Counter = Counter()
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        # total=None because streaming can't size ahead; tqdm still gives a rate.
        for row in tqdm(rows, desc="filtering FinReflectKG", unit="triple"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_ticker[row["ticker"]] += 1
            by_extraction_type[row.get("extraction_type", "<missing>")] += 1
            by_ticker_year[(row["ticker"], row["year"])] += 1
            count += 1
    return count, by_ticker, by_extraction_type, by_ticker_year


def write_report(
    count: int,
    by_ticker: Counter,
    by_extraction_type: Counter,
    by_ticker_year: Counter,
    report_path: Path,
) -> None:
    lines = [
        "# FinReflectKG Subset Report",
        "",
        f"**Source:** HuggingFace `{DATASET_ID}` (streaming)",
        f"**Filter:** ticker ∈ {sorted(TARGET_TICKERS)}, year ∈ {sorted(TARGET_YEARS)}",
        f"**Output:** `{OUT_PATH.relative_to(REPO_ROOT)}`",
        f"**Total triples kept:** {count:,}",
        "",
        "## Triples per ticker",
        "",
        "| Ticker | Triples |",
        "|---|---:|",
    ]
    for ticker in sorted(TARGET_TICKERS):
        lines.append(f"| {ticker} | {by_ticker.get(ticker, 0):,} |")
    lines += [
        "",
        "## Triples per ticker × year",
        "",
        "| Ticker | 2022 | 2023 | 2024 |",
        "|---|---:|---:|---:|",
    ]
    for ticker in sorted(TARGET_TICKERS):
        row = [ticker]
        for year in sorted(TARGET_YEARS):
            row.append(f"{by_ticker_year.get((ticker, year), 0):,}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Extraction type breakdown",
        "",
        "FinReflectKG's `extraction_type` field distinguishes triples whose "
        "dates were LLM-extracted (`explicit`, `relative`) from those that "
        "default to the filing year (`default`). This matters for our baseline "
        "comparison — TempoRAG-KG's value proposition is precisely that our "
        "explicit-extracted intervals outperform default-filing-year fallback.",
        "",
        "| extraction_type | Triples |",
        "|---|---:|",
    ]
    for kind, n in by_extraction_type.most_common():
        lines.append(f"| {kind} | {n:,} |")
    lines += [
        "",
        f"**Default rate:** "
        f"{by_extraction_type.get('default', 0) / max(count, 1):.1%} of triples use the filing-year default.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file already exists.",
    )
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"{OUT_PATH} already exists — pass --force to re-download.")
        return

    print(f"Streaming {DATASET_ID} and filtering...")
    rows = stream_filtered(DATASET_ID)
    count, by_ticker, by_extraction_type, by_ticker_year = write_jsonl(rows, OUT_PATH)
    print(f"\nWrote {count:,} triples to {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Per-ticker: {dict(by_ticker)}")
    print(f"extraction_type: {dict(by_extraction_type)}")
    write_report(count, by_ticker, by_extraction_type, by_ticker_year, REPORT_PATH)
    print(f"Wrote report to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
