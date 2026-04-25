# A3 — 10-K Acquisition & Parsing Report

**Task:** Download + parse 25 10-K filings (AAPL, MSFT, GOOGL, AMZN, META × FY2019–FY2023), produce chunk-ready text, flag any failure, estimate chunk count vs the [docs/10k_scoping.md](10k_scoping.md) projection.
**Status:** ✅ Complete (25/25 downloaded, 25/25 parsed, 0 silent skips).
**Date:** 2026-04-17.

## 1. Acquisition

- **Script:** [scripts/download_10k.py](../scripts/download_10k.py)
- **Manifest:** [data/10k/manifest.json](../data/10k/manifest.json)
- **Failures log:** [data/10k/failures.jsonl](../data/10k/failures.jsonl) (empty)
- **Result:** 25/25 filings downloaded (~87 MB total HTML). All accession numbers + SHA256 + period-of-report recorded in the manifest.
- **EDGAR compliance:** User-Agent identifies the researcher (`bsupanutkom@gmail.com`); rate limit 5 req/s (half the SEC cap); 3-attempt exponential backoff on 429/5xx.

## 2. Parsing

- **Script:** [src/parse_10k.py](../src/parse_10k.py)
- **Tests:** [tests/test_parse_10k.py](../tests/test_parse_10k.py) (4 tests, including AAPL FY2023 end-to-end)
- **Parse manifest:** [data/10k/parse_manifest.json](../data/10k/parse_manifest.json)
- **Parse failures log:** [data/10k/parse_failures.jsonl](../data/10k/parse_failures.jsonl) (empty)
- **Target sections:** Item 1, 1A, 7, 7A, 8 (per [docs/10k_scoping.md §2](10k_scoping.md))
- **Boilerplate removed:** cover page + TOC (via second-occurrence-of-"Item 1" heuristic), standalone page numbers, Items 2–6/9–16, exhibits, signatures.
- **Result:** 25/25 filings parsed successfully. Every filing contains all five target sections above the 500-char presence threshold; `missing_items` is empty across the board.

## 3. Chunk count — actual vs projection

Tokenized section-wise with `tiktoken` (`cl100k_base`), chunk size 512, overlap 100.

| | Planned | Actual | Δ |
|---|---|---|---|
| Chunks | 2,750 | **3,810** | **+1,060 (+38.5%)** |
| Tokens | — | 1,556,636 | — |
| Avg chunks / filing | 110 | 152.4 | +38.5% |

Raw per-filing breakdown in [data/10k/chunk_estimate.json](../data/10k/chunk_estimate.json).

### Where the overshoot comes from

Chunk count scales with total tokens; the projection under-estimated how large mega-cap filings are.

| Ticker | Avg tokens/filing | Avg chunks/filing |
|---|---|---|
| AAPL | 40.2k | 99 |
| AMZN | 50.0k | 122 |
| GOOGL | 61.2k | 150 |
| META | 74.6k | 183 |
| MSFT | **85.3k** | **208** |

MSFT is ~2× AAPL; META ~1.9×. This is real content (Item 8 notes are substantially longer for MSFT/META), not boilerplate we failed to scrub.

## 4. Cost implication

Gemini 1.5 Flash extraction: ~$0.00028 / chunk (per [docs/10k_scoping.md §4](10k_scoping.md)).

| Scenario | Chunks | Cost |
|---|---|---|
| Single-pass extraction | 3,810 | **$1.07** |
| + 2× re-run buffer (typical for prompt iteration) | 7,620 | **$2.14** |

The v2 plan budgeted **$2** for KG build — the 2× buffer now puts us **$0.14 over** that line item. Options, in order of how much I'd recommend:

1. **Do nothing, re-baseline the line item to $2.20.** Total committed budget was ~$12.55 of $20; we have slack. Recommended.
2. Reduce chunk size to 384 (would *increase* count; wrong direction) — skip.
3. Drop Item 8 for the smallest cap(s) — would lose the densest temporal content. Skip.
4. Cap to 1 re-run buffer instead of 2 ($1.07 → $1.50 tolerance). Viable if prompt is stable after P1 pilot.

Recommendation: keep the $2 ceiling as *per-run* rather than *per-iteration*, and re-check after the P1 pilot (20 chunks) tells us how stable the extraction prompt is.

## 5. Bugs encountered (so future-me knows)

1. **AAPL 404s on `www.sec.gov/Archives/...`.** Session had `Host: data.sec.gov` header leaking into cross-subdomain requests. Fix: drop the Host header, let `requests` derive per-URL.
2. **MSFT/GOOGL/META FY2019/2020 "no filing found".** Older filings don't appear in `filings.recent` — they live in paginated `filings.files[]` archives. Fix: walk archives when `recent` misses. Both shapes are column-oriented with the same keys, so one iterator handles both.
3. **`FeatureNotFound: lxml`.** Missing dep; added `lxml>=4.9` to `requirements.txt`.
4. **`_stats.json` write crashed under pytest tmp_path.** `Path.relative_to(REPO_ROOT)` throws when the output dir is outside the repo. Fix: `_rel_or_abs()` falls back to absolute string.

## 6. What's next (feeds P1)

- Sections are chunk-ready at [data/10k/sections/<ticker>/FY<year>/item_<N>.txt](../data/10k/sections/).
- Next task **A4**: write `src/chunk_10k.py` that consumes these section files, emits chunk JSONL with `(chunk_id, ticker, fy, item, token_count, text, filing_date, period_of_report)` metadata wired from `data/10k/manifest.json`. Temporal metadata on the chunk is essential — it's what lets the KG attach `valid_from` / `valid_to` deterministically to extracted triples.
- Then **P1 pilot**: sample 20 chunks, run Gemini Flash extraction, sanity-check triples before committing to the full 3,810-chunk run.
