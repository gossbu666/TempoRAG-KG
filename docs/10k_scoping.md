# 10-K Scoping Note (K5)

**Date:** 2026-04-17
**Status:** Draft for K1 (v2 plan) input
**Author:** Supanut (team lead), for team review

## 1. Scope

| Dimension | Value |
|---|---|
| Companies | 5 (AAPL, MSFT, GOOGL, AMZN, META) |
| Fiscal years | 2019, 2020, 2021, 2022, 2023 |
| Total filings | 25 |
| Filing type | Annual report on Form 10-K (not 10-Q, not 8-K) |
| Source | SEC EDGAR (free, public) |
| Local storage | `data/10k/raw/<ticker>/<fy>.html` |

### CIKs for EDGAR retrieval

| Ticker | CIK |
|---|---|
| AAPL | 0000320193 |
| MSFT | 0000789019 |
| GOOGL | 0001652044 |
| AMZN | 0001018724 |
| META | 0001326801 |

## 2. Section selection

A 10-K is ~100–150 pages. Processing every page would blow the budget and the signal-to-noise ratio. We process only temporal-rich sections:

| Section | Why include | % of filing |
|---|---|---|
| **Item 1 — Business** | Segment descriptions, product launch years, acquisitions | ~15% |
| **Item 1A — Risk Factors** | Evolves year-over-year; good for longitudinal KG | ~15% |
| **Item 7 — MD&A** | **Primary temporal signal** — YoY comparisons, guidance, "during fiscal 2022..." | ~20% |
| **Item 7A — Quant/Qual Market Risk** | Forward-looking temporal statements | ~3% |
| **Item 8 — Financial Statements Notes** | Event dates, acquisition/divestiture timing, lease maturities | ~15% |

**Skipped:** cover page, Item 1B (legacy staff comments), Items 2–6, Item 9A (controls boilerplate), Part III (governance, mostly static references to proxy), Part IV (exhibits).

Effective ratio: **~50% of each filing processed.**

## 3. Chunk strategy

Reuse v1 convention from KG²RAG (Zhu et al., 2025):

- **Chunk size:** 512 tokens
- **Overlap:** 100 tokens
- **Tokenizer:** tiktoken `cl100k_base` (GPT-4 encoder — matches typical LLM tokenization; we store chunks as raw text, so the exact tokenizer only affects count, not downstream behavior)
- **Section-aware splitting:** never cross a section boundary within a chunk; emit short final chunk if section ends before 512 tokens reached

Per-filing chunk estimate:
- Avg filing: ~60,000 words ≈ ~90,000 tokens (1.5 tokens/word for dense financial prose)
- Processed portion (~50%): ~45,000 tokens
- Chunks per filing (512 tokens, 100 overlap → ~412 effective): **~110 chunks**
- **Total corpus: 25 × 110 ≈ ~2,750 chunks**

## 4. Cost projection (Gemini 1.5 Flash)

Pricing (as of 2025 pricing page; verify with @context7 or Gemini docs before running):
- Input: **$0.075 / 1M tokens** (context ≤ 128K)
- Output: **$0.30 / 1M tokens**

Per-chunk extraction call estimate:
- Input: prompt template (~600 tok) + chunk (~512 tok) + schema reminder (~200 tok) = **~1,300 tok**
- Output: JSON triples, financial 10-K prose is triple-rich → **~600 tok** (20% safety margin over textbook ~500)

Cost per chunk:
- Input: 1,300 × $0.075 / 1M = **$0.0000975**
- Output: 600 × $0.30 / 1M = **$0.00018**
- **Total: ~$0.00028 / chunk**

Projected extraction cost:
| Scope | Chunks | Cost |
|---|---|---|
| Pilot (20 chunks, 1 filing section) | 20 | ~$0.006 |
| Full corpus, single pass | ~2,750 | **~$0.77** |
| Full corpus + 1 re-run buffer | ~5,500 | ~$1.54 |
| Hard cap for KG build phase | — | **$2.00** |

Leaves ~$18 of $20 budget for eval-side LLM calls (GPT-4o-mini, GPT-4o ceiling run, RQ4 sweep). Matches v1 allocation envelope.

## 5. Does 25 filings produce enough KG for multi-hop temporal QA?

Back-of-envelope on triple density:

- KG²RAG paper reports ~15–30 triples per 512-token chunk on Wikipedia prose
- Financial prose is triple-denser (every number has an entity + metric + period) — estimate **25–40 triples/chunk**
- Corpus estimate: 2,750 chunks × 30 triples = **~82,500 triples**
- After dedup (same entity-relation across years): ~40,000–50,000 unique triples

**Temporal coverage check:**
- 5 years of coverage per company → multi-hop questions like "who was Apple's CFO in 2020 vs 2023" become natural
- 5 companies × 5 years = 25 time-points; sufficient for RQ4 *inverse-scaling* story at temporal-subset level
- **Limitation:** only intra-company temporal reasoning; cross-company temporal QA ("which tech mega-cap first reported AI revenue") is stretch territory but possible

## 6. Evaluation question source (still open)

We have the KG. We still need **questions** with **gold temporal answers**. Options:

| Option | Pros | Cons |
|---|---|---|
| **FinanceBench** (Patronus AI, 2024) | Purpose-built, 150 human-annotated Q on public filings | May not overlap our 5 tickers / 5 years |
| **FinQA** (Chen et al., 2021) | Large, numerical reasoning over 10-K tables | Not strongly temporal |
| **Synthesize our own from templates** | Full control over temporal structure; matches our 5×5 | Manual effort; IAA needed (2 annotators on 100 Q) |
| **Hybrid** (FinanceBench subset + synthesized) | Best coverage | Most work |

**Recommendation:** Start with **FinanceBench subset** (filter to our 5 tickers, expect ~30–50 matching Q). If < 100 Q, synthesize the remainder using templates like:
- `"What was <ticker>'s <metric> in <year>?"`
- `"How did <ticker>'s <segment> revenue change from <year_a> to <year_b>?"`
- `"Who was the <role> at <ticker> in <year>?"` (multi-hop through named executive officers table)

Defer synthesis protocol to v2 plan.

## 7. Acquisition plan

Small script `scripts/download_10k.py`:

1. Read ticker→CIK map (hardcoded, 5 entries)
2. For each (ticker, fy): query EDGAR submissions index → find 10-K with period-of-report ending in that FY → get primary document URL
3. Download HTML to `data/10k/raw/<ticker>/<fy>.html`
4. Log: URL, filing date, document length, SHA256
5. **No paid API** — EDGAR is free

Then `src/parse_10k.py`:

1. Parse HTML → extract Items 1, 1A, 7, 7A, 8 (section headers via regex / BeautifulSoup)
2. Store sections as plain text in `data/10k/sections/<ticker>/<fy>/<item>.txt`
3. Emit `data/10k/manifest.json` with counts

Expected effort: 1 working day including debugging SEC HTML quirks.

## 8. Go/no-go verdict for this scope

**GO.** Reasoning:
- Cost projection (~$0.77 for KG build) is 6% of budget — very safe
- Chunk count (~2,750) is tractable at Gemini Flash latency (~1s/chunk → ~45 min wall-clock single-threaded; parallelize if needed)
- Temporal density matches RQ4 story requirements
- All data is public, no licensing issues
- Tooling (SEC EDGAR, BeautifulSoup, tiktoken) is all free and well-documented

**Conditions to re-check before running:**
1. Gemini 1.5 Flash pricing unchanged (check Context7 or Google docs before pilot)
2. Rate limits on Gemini Flash free tier allow ~2,750 calls (paid tier bypasses this)
3. EDGAR HTML structure hasn't changed for any of the 5 tickers (spot-check 1 filing manually)

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Some 10-K HTML has unusual section markup (SEC allows flexibility) | Medium | Low | Robust parser with per-filing fallback; log parse failures for manual cleanup |
| Temporal QA coverage from FinanceBench < 50 Q for our 5 tickers | Medium | Medium | Synthesis templates as backup (§6) |
| Financial triples extracted have too many numerical/quantitative entities that don't form multi-hop graph | Low | Medium | Pilot on 20 chunks (v2 T6) validates triple shape before full build |
| Gemini Flash output format drifts on financial jargon | Low | Medium | Pilot catches this; prompt iteration in K1's Phase 1 adaptation |

## 10. Next steps (handoff to K1)

K1 (tasks/plan.md v2) should encode:

- **Phase 0 carry-over:** v1 infra stays (cache, eval, IAA — drop α threshold; sampling — needs adaptation for 10-K chunks)
- **Phase 1 adaptation:**
  - T1' Adapt `src/sampling.py` to chunk-level sampling for 10-K instead of question-level sampling for HotpotQA
  - T5' Rewrite `prompts/extract_v1.txt` for financial domain (replace Wikipedia examples with 10-K examples)
- **Phase 2 pilot:** same gate structure, different data source
- **Phase 3 full KG build:** 25 filings × ~110 chunks each, hard cap $2
- **Phase 4 pipeline:** same temporal filter / GEAR / GoG structure (no changes needed — KG is KG)
- **Phase 5 evaluation:**
  - Question source: FinanceBench subset + synthesized (see §6)
  - **RQ4 is the spine** (was optional in v1)
  - RQ2 and RQ3 secondary
  - RQ1 failure taxonomy same 5 types, re-interpreted for financial context
- **Phase 6 progress report:** target 2026-05-15 (4 weeks out)

Budget re-forecast:
| Phase | Est. cost |
|---|---|
| Pilot (20 chunks) | $0.01 |
| Full KG build | $1.54 (with buffer) |
| RQ3 extraction eval | $0 (reuses cache) |
| RQ2 ablation (3 conditions × question set) | ~$2 |
| RQ4 generator sweep (4 generators × temporal subset) | ~$3 |
| GPT-4o ceiling run (optional) | ~$6 |
| **Total committed** | **~$12.55** |
| **Buffer** | **~$7.45** |
