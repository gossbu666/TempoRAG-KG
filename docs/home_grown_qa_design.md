# Home-grown QA set (A4b) — Design note

**Target:** 50 hand-authored temporal QA pairs, complementing
`data/qa/multihop_filtered.jsonl` (79 Qs from FinReflectKG-MultiHop).

**File:** `data/qa/home_grown.jsonl`

## Why home-grown exists

MultiHop-555 (filtered to 79) skews:

| Axis | MultiHop coverage | Gap home-grown fills |
|---|---|---|
| Tickers | INTC (24) / CSCO (15) / ORCL (13) heavy; **AMZN = 0** | AMZN-only bucket (10 Qs) |
| Years | 2022–2024 only | Mag5 FY2019–2021 retro bucket (15 Qs) |
| Scope | cross_company = 6 (of 79) | Dedicated cross_company bucket (18 Qs) |
| Edge cases | 0 fiscal-vs-calendar disambiguation | fiscal_vs_calendar bucket (5 Qs) |
| Edge cases | 0 forward_looking | forward_looking bucket (2 Qs) |

Gap analysis script: `data/qa/multihop_filtered.jsonl` scoped vs
`data/samples/10k_chunks.jsonl` ticker-FY pairs — 22 of 45 pairs (49%)
have zero MultiHop coverage. Home-grown fills the high-value ones.

## Schema

JSONL, one question per line. Fields mirror MultiHop where shared, adds
home-grown-specific fields.

```json
{
  "question_id": "H001",
  "question": "...",
  "answer": "...",
  "hop_count": 2,
  "scope": "cross_company | inter_year | intra | fiscal_vs_calendar | forward_looking",
  "category": "revenue | segment | headcount | ...",
  "tickers": ["AAPL", "MSFT"],
  "years": [2022],
  "verification_hint": "AAPL FY2022 item 7 vs MSFT FY2022 item 7",
  "source_dataset": "home_grown_v1"
}
```

**Differences from MultiHop schema:**

- `question_id` — string prefix `H` so it's visually distinct from MultiHop numeric ids.
- `source_chunks` / `evidence` omitted at draft time; backfilled after KG build when we can match answer to chunk.
- `verification_hint` added — human-readable pointer to the filing item to eyeball the ground truth quickly.

## Bucket design (50 Qs total)

### B1. `cross_company` (18 Qs)

- Same metric, same fiscal year, two tickers. Tests whether retrieval
  can cross company boundaries in the KG.
- Pairs chosen for known-difference direction (so F1/EM is meaningful):
  AAPL vs MSFT FY2022 revenue (AAPL 2×), GOOGL vs META FY2023 ad revenue,
  NVDA vs INTC FY2024 data center, AMZN AWS vs MSFT Azure (structurally).

### B2. `inter_year` Mag5 retro (15 Qs)

- Same ticker, two fiscal years within FY2019–2021. Covers the blind
  spot MultiHop has (it only goes 2022+).
- Useful for prompting tests on fiscal-year identity (AAPL FY2020 = the
  pandemic year; MSFT FY2020 = July 2019 – June 2020).

### B3. `intra` AMZN-only (10 Qs)

- Since MultiHop has zero AMZN Qs, this bucket proves our pipeline
  doesn't fail catastrophically on a ticker the training data never saw.
- Mix of single-year fact lookup (FY2023 North America segment revenue)
  and intra-filing reasoning (segment operating income vs consolidated).

### B4. `fiscal_vs_calendar` (5 Qs)

- Explicitly tests the fiscal-year disambiguation rule in
  `prompts/extract_v1.txt`. A question like "What was Microsoft's
  revenue for the fiscal year ending June 2023?" should return the
  FY2023 filing answer ($211.9B), not the calendar-year 2023 synthesis.
- Covers MSFT (fiscal end June), ORCL (fiscal end May), CSCO (fiscal
  end late July), ADBE (fiscal end early December).

### B5. `forward_looking` (2 Qs)

- Tests that the KG preserves `temporal_type: forward_looking` and
  that retrieval doesn't mistake a commitment for a realized fact.

## Verification discipline

- Every answer cross-checked against `data/samples/10k_chunks.jsonl`
  (the same chunks feeding the KG).
- Numbers cite the filing's own dollars-in-millions convention to match
  what the KG will store.
- For cross-company, explicit "as of" anchor so the answer is
  well-defined even under year-ambiguity.

## Known limitations

- Home-grown questions are authored by the extraction team → risk of
  bias toward questions our KG happens to answer. Mitigated by:
  (a) writing questions BEFORE seeing the KG output (this task runs
  while extract is still live), (b) drawing ground truth from raw
  chunks not KG triples, (c) fiscal-vs-calendar bucket specifically
  stress-tests a known weakness.
- Authored by one person in one sitting → no inter-rater check. Small
  scale (50 Qs) makes single-author acceptable; a second teammate can
  review before eval.
