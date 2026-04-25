# TempoRAG-KG v2 — Implementation Plan (Post-TA-Pivot)

**Last updated:** 2026-04-19 (v2.2 — 10-ticker expansion)
**Status:** Phase 0 partially stale — ticker set expanded to 10 tech (2026-04-19); A1/A3 need re-run for 5 new tickers + FY2024. Eval harness complete; blocked on teammate prompt review before P1 pilot.
**Repo:** https://github.com/gossbu666/TempoRAG-KG
**Related docs:**
- `docs/ta_consultation_2026_04_16.md` — full TA meeting record
- `docs/ta_feedback_for_team_2026_04_16.md` — English summary for team
- `docs/10k_scoping.md` — sector scope, chunk strategy, cost projection
- `docs/temporal_methods_scan.md` — lit scan justifying extract+use as contribution (K4)
- `docs/a3_report.md` — 10-K download + parse + chunk-count writeup (A3 delivered)
- `docs/prompt_review.md` — open teammate review on `prompts/extract_v1.txt`
- `tasks/archive/plan_v1_hotpotqa.md` — previous (v1) plan, superseded
- `tasks/todo.md` — live checklist

**If this document and memory disagree, this document wins.**

---

## 1. What changed from v1

See `docs/ta_consultation_2026_04_16.md` for the full meeting record. Short version:

| Dimension | v1 (HotpotQA) | v2 (10-K) |
|---|---|---|
| Primary dataset | HotpotQA + MuSiQue (Wikipedia) | 10-K SEC filings (tech mega-caps, 10 tickers) |
| Corpus size | 1,000 + 500 questions | 30 filings (10 tickers × FY2022-2024), ~4,600 chunks projected |
| Primary RQ | RQ2 (F1 lift from temporal tagging) | **RQ4 (small-vs-large capability parity)** |
| Temporal proof | Empirical ±1 year justification | **Dropped** — AI is probabilistic, not meaningful |
| IAA target | α ≥ 0.70 hard target | **Dropped** — magic number; report α without threshold |
| Framing | Cost reduction | **Capability** (cost is a by-product) |
| Contribution | Temporal intervals in KG edges | **Extraction + usage** of temporal info (storage is not novel) |

## 1b. Prior work positioning

Research in 2026-04-18/19 surfaced four papers that must be defended against explicitly. Memory `project_prior_work_landscape.md` carries the full comparison; the table below is the TL;DR.

| Paper | What they did | How we differ |
|---|---|---|
| **KG²RAG** (NAACL 2025) | KG-guided retrieval on Wikipedia, eval on HotpotQA | We port the retrieval mechanism to 10-K and add edge-level `[valid_from, valid_to]` |
| **FinReflectKG** (arxiv 2508.17906) | Reflection-agent KG extraction from 743 S&P 100 10-Ks | Their temporal is "Month YYYY" strings, mostly filing-year default (`extraction_type="default"`, **76.6%** of our 10-ticker subset). Ours is explicit-extraction temporal on 10 tech tickers. Used as **baseline KG arm** in ablation. |
| **FinReflectKG-MultiHop** (arxiv 2510.02906) | 555-Q multi-hop benchmark over their KG; KG retrieval beats page-window +24% LLM-Judge | Their inter-year scope is the weakest (6.72 vs 7.47 intra-doc LLM-Judge) — **that's the gap we target**. They don't isolate KG from temporal; we do (L1/L2/L3). |
| **TA-RAG** (arxiv 2507.22917) | Timestamped chunk metadata (no KG) + interval filter | **This is our L1 citation** — shows temporal alone can lift vanilla RAG by +27.5–28.1pp without any KG. |

**Our differentiation one-liner:**

> TempoRAG-KG = KG²RAG retrieval backbone + FinReflectKG 10-K domain + **explicit edge-level temporal intervals** (not filing-year default) + **L1/L2/L3 layered ablation** that separates KG contribution from temporal contribution + **RQ4 cost-vs-capability story** (unique to us).

## 1c. Three-layer justification chain

Before claiming "Temporal + KG + RAG" is the right design, the progress report must walk up the justification ladder independently.

| Layer | Claim | Ablation arm | Citation we lean on |
|---|---|---|---|
| **L1** | Temporal filtering helps RAG even without a KG | Vanilla RAG vs **Temporal-Vanilla-RAG** (chunk metadata filter) | TA-RAG (arxiv 2507.22917) |
| **L2** | KG contributes over plain RAG | Vanilla RAG vs **KG²RAG** | KG²RAG (NAACL 2025) |
| **L3** | KG + Temporal beats each alone | All-of-the-above vs **TempoRAG-KG (Full)** | Our contribution |

**Why this matters:** the reviewer's first move will be "did you need all three?". Combining interventions without layered ablations is the classic refereeing red flag. Our 4-arm matrix (Vanilla / Temporal-Vanilla / KG²RAG / Full) answers it directly.

## 1d. Locked design decisions (2026-04-18/19)

Decisions frozen this sprint — changing any of them requires re-scoping cost + re-running projections.

| # | Decision | Rationale |
|---|---|---|
| 1 | Triple temporal format: `int year` (valid_from, valid_to) | Matches current prompt + tests; ISO 8601 + precision flag deferred to future work |
| 2 | Metrics: **LLM-Judge (0-10) + BERTScore + cost-per-Q**, NOT F1/EM | 10-K answers are paragraph-length; F1/EM undercounts. Judge matches FinReflectKG-MultiHop for comparability. |
| 3 | QA source: **hybrid** — filtered FinReflectKG-MultiHop 555 → **79 Qs** over our 10 tickers (intra 35 / inter_year 38 / cross_company 6) + ~50 home-grown temporal edge cases | Pure home-grown is unreproducible; pure MultiHop under-tests edge cases. 10-ticker expansion unlocked the cross_company scope (was 0 under Mag5). |
| 3b | Ticker set: **10 tech-megacap** (AAPL/MSFT/GOOGL/AMZN/META + CSCO/ORCL/INTC/NVDA/ADBE) × FY2022-2024 | Chosen 2026-04-19 to close the cross_company scope gap while keeping prompt domain coherent (all 10 are NASDAQ 100 tech/comm — same 10-K structure, no prompt re-tune needed). See `data/qa/multihop_filtered.jsonl` + `docs/multihop_filter_report.md`. |
| 4 | Extraction: single-pass (KG²RAG parity), **not** reflection | Budget + comparability to KG²RAG backbone |
| 5 | Tables: **future work** — v2 text-only extraction | Same scope as KG²RAG; avoids docling dependency |
| 6 | XBRL: **ground-truth validation only** (not primary data source) | Narrative facts XBRL can't cover are what multi-hop needs |
| 7 | Hallucination guard: evidence must be substring of chunk.text | Stronger than KG²RAG's post-filter (whole sentence vs single token) — **shipped 2026-04-19** |

## 2. Current state snapshot

| Category | Status |
|---|---|
| EDA on HotpotQA/MuSiQue (35% / 38.3% temporal) | ✅ Done, kept as motivation |
| Proposal v1 submitted (2026-03-23) | ✅ Done |
| TA feedback (2026-04-16) | ✅ Documented |
| v1 plan/todo archived | ✅ Done |
| v2 plan (this file) | ✅ v2.1 updated 2026-04-19 (post-landscape research) |
| `src/cache.py` + tests | ✅ Done (reusable as-is) |
| `src/eval.py` + tests | ✅ Done for F1/EM; **needs upgrade** to LLM-Judge + BERTScore |
| `src/iaa.py` + tests | ✅ Done (α reported without threshold) |
| `src/sampling.py` + `sample_10k_chunks` | ✅ A1 delivered for Mag5 FY2019-2023 (3,810 chunks) — **needs re-run** for 5 new tickers + FY2024 |
| `src/parse_10k.py` + tests | ✅ A3 delivered for Mag5 (25/25 filings × 5 items) — **needs extension** to 10 tickers × FY2022-2024 (20 more filings) |
| `scripts/download_10k.py` | ✅ A3 delivered for Mag5 — **needs re-run** for 5 new tickers + FY2024 |
| `prompts/extract_v1.txt` (financial) | ✅ A2 drafted; **awaiting teammate review** (`docs/prompt_review.md`) — same prompt covers 10 tech tickers (no domain drift) |
| `prompts/archive/extract_v1_wikipedia.txt` | ✅ v1 prompt archived |
| `src/kg_extract.py` + tests (zero-cost Python) | ✅ delivered + **hallucination guard added 2026-04-19** (28 tests passing) + DEFAULT_COMPANY_NAMES extended to 10 tickers |
| 10-K scoping note | ✅ Done (`docs/10k_scoping.md`) — **needs update** for 10-ticker budget re-projection |
| Team pivot sync | ✅ Agreed (per user, 2026-04-17) |
| Landscape memory + prior-work comparison | ✅ Done (`memory/project_prior_work_landscape.md`, 2026-04-19) |
| Temporal methods lit scan | ✅ Done (`docs/temporal_methods_scan.md`, K4) |
| English TA summary for team | ⏳ Pending (K2 optional; ta_consultation doc may suffice) |
| **Prompt review sign-off** | 🔴 **BLOCKING** — nobody assigned yet |
| FinReflectKG HF subset download (baseline arm) | ✅ **Done for 10 tickers (74,979 triples, 76.6% default rate)** — one coverage gap: META FY2022 = 0 triples in the HF release |
| Hybrid QA set assembly | ⏳ MultiHop filter DONE (79 Qs, `data/qa/multihop_filtered.jsonl`); home-grown ~50 Qs still to author (A4b) |
| Eval harness upgrade (LLM-Judge + BERTScore) | ✅ Done — `score_with_judge`, `score_bertscore`, `aggregate_by_scope` + 36 tests (T3.2) |
| Retrieval pipeline arms (G1-G4) | ⏳ Pending — largest remaining work item |

## 3. Operating constraints

- **Budget:** $20 hard cap. Current spend: **$0**. Re-baselined projection under **10-ticker scope** (~4,600 chunks projected from 30 filings × ~152 chunks/filing observed in A1): **~$2.65 for B1 pilot + full build** (single-pass + 10% re-run buffer), leaving ~$15 for eval + judge. ~20% increase vs Mag5-only; still well under cap. See `docs/10k_scoping.md` §10 (to be updated).
- **Deadline:** Progress report **2026-05-15** (~4 weeks from TA pivot)
- **Cache-first:** every paid API call via `src.cache.Cache` (already enforced in `src/kg_extract.py`)
- **Free tier priority:** Groq LLaMA-3.1-8B is the primary generator; Gemini 1.5 Flash for extraction + LLM-Judge
- **Pilot gate:** no full KG build until 20-chunk P1 pilot passes GO
- **Prompt review gate:** no paid API call until `docs/prompt_review.md` signed off
- **Seed:** `RANDOM_SEED=42` everywhere
- **Hard cost guards** on every paid script (abort if exceeded)
- **Hallucination guard:** enforced in `src/kg_extract._validate_triple` (evidence must appear in chunk text)

## 4. Story spine (what the progress report argues)

> **Small models augmented with temporal-KG retrieval can match or close the gap with larger models on temporally-grounded multi-hop financial QA — and each of the three layers (Temporal, KG, Temporal+KG) contributes independent lift.**
>
> Our evidence (4-arm × 2-generator matrix):
> | Arm | Retriever | Temporal filter | Cites |
> |---|---|---|---|
> | Vanilla | BM25/dense | — | baseline |
> | Temporal-Vanilla-RAG (L1) | BM25/dense | chunk-metadata filter | TA-RAG |
> | KG²RAG (L2) | KG-guided chunk expansion | — | KG²RAG (NAACL 2025) |
> | TempoRAG-KG Full (L3) | KG-guided + edge-level `[valid_from, valid_to]` filter | **ours** |
>
> × 2 generators {Groq LLaMA-3.1-8B, GPT-4o-mini} × stratified scope {intra-doc, inter-year, cross-company} = **24 cells**.
>
> **Primary metric:** LLM-Judge 0-10 + BERTScore. **Primary comparison:** Δ(Judge) per generator for RQ4; arm-level lift (L1, L2, L3) for the ablation.

Secondary contributions:
- Temporal failure taxonomy (5 types: Stale / Conflict / Missing / Relative / Hop-failure) re-interpreted for financial context
- Test scenarios showing failure→fix cases (TA feedback #4)
- Comparison against **FinReflectKG** (default filing-year temporal) on same 10 tickers — quantifies the value of explicit-extraction temporal over metadata-default

## 5. Dependency graph

```
Phase 0 — Adaptation  [partially stale after 10-ticker expansion 2026-04-19]
  ├─ A1 Adapt src/sampling.py for 10-K chunk sampling          ⚠️ needs re-run (5 new tickers + FY2024)
  ├─ A2 Rewrite prompts/extract_v1.txt for financial domain    ✅ DRAFT (review pending; no rewrite for new tickers — same domain)
  ├─ A3a 10-K acquisition + HTML section parser (Mag5)         ✅ DONE (25/25 filings FY2019-2023)
  ├─ A3b 10-K acquisition extension (new 5 + FY2024)           ⏳ pending  — 20 more filings to download
  ├─ A4a MultiHop filter (555 → 79 Qs over 10 tickers)         ✅ DONE (`data/qa/multihop_filtered.jsonl`, 2026-04-19)
  ├─ A4b Home-grown QA (~50 Qs, cross-company + inter-year)    ⏳ pending
  ├─ A5 Download FinReflectKG HF subset — baseline arm         ⏳ running 2nd pass (10 tickers, job b43mjr50m)
  └─ A6 Hallucination guard in kg_extract                      ✅ DONE (2026-04-19)
          │
          ▼
    ╔══════════════╗
    ║ CHECKPOINT 0 ║  prompt review signed off, QA set ≥100, FinReflectKG subset on disk
    ╚══════════════╝
          │
          ▼
Phase 1 — Pilot  (2 days)
  ├─ P1 Pilot extraction on 20 chunks (hard $0.20 cap)
  └─ P2 Pilot report + GO/NO-GO
          │
          ▼
Phase 2 — KG build + Eval harness + Gold annotation  (1 week, parallel)
  ├─ B1 Full KG build over ~4,600 chunks (hard $2.65 cap)
  ├─ B2 FinReflectKG baseline KG on same 10 tickers (zero API cost — HF subset)   [NEW role]
  ├─ B3 Gold annotations (50 passages, 2 annotators, α reported without threshold)
  └─ T3.2 Upgrade src/eval.py — LLM-Judge (Gemini Flash as judge) + BERTScore    [NEW]
          │
          ▼
Phase 3 — Pipeline (4 retrieval arms)  (1 week)
  ├─ G1 Edge-level temporal filter (interval overlap; tolerance=0 default)
  ├─ G2 Vanilla RAG (BM25/dense over chunks)
  ├─ G3 Temporal-Vanilla-RAG (L1 — chunk-metadata filter, TA-RAG style)          [NEW ARM]
  ├─ G4 KG²RAG (L2 — KG-guided expansion, no temporal)
  ├─ G5 TempoRAG-KG Full (L3 — G4 + edge-level temporal filter)
  └─ G6 End-to-end integration smoke test (10 Qs across all 4 arms)
          │
          ▼
    ╔══════════════╗
    ║ CHECKPOINT 1 ║  all 4 arms answer 10-Q smoke set; temporal filter measurably affects ≥3 Q
    ╚══════════════╝
          │
          ▼
Phase 4 — Evaluation  (1 week, parallel)
  ├─ E1 RQ4 primary sweep — 4 arms × 2 generators × stratified scope (hard $5 cap)
  ├─ E2 RQ3 extraction accuracy vs 50-passage gold
  ├─ E3 RQ1 failure taxonomy on 50 errors
  ├─ E4 Test scenarios (10 failure→fix cases, TA feedback #4)
  └─ E5 XBRL ground-truth validation on numeric triples (edgartools; zero API)   [NEW]
          │
          ▼
    ╔══════════════╗
    ║ CHECKPOINT 2 ║  L1/L2/L3 lift numbers + RQ4 Δ-Judge with bootstrap CI
    ╚══════════════╝
          │
          ▼
Phase 5 — Progress Report  (4 days)
  └─ R1 Write progress_report.tex
```

**Timeline:** 4 weeks total if Phase 2 & 4 parallelize across 4 teammates. Phase 3 retrieval pipeline (G1-G6) is the single largest remaining work item (~1 person-week).

## 6. Task details

### Phase 0 — Adaptation

#### A1 — Adapt `src/sampling.py` for 10-K  ✅ DELIVERED

- **Goal:** Chunk-level deterministic sampling over the 10-K corpus.
- **Status:** Done. `sample_10k_chunks()` produces `data/samples/10k_chunks.jsonl` — **3,810 chunks** (byte-identical on re-run). Chunk record includes `{chunk_id, ticker, fy, item, text, sha256, token_count, filing_date, period_of_report}`.
- **Files:** `src/sampling.py`, `tests/test_sampling.py`, `data/samples/10k_chunks.jsonl`

#### A2 — Rewrite `prompts/extract_v1.txt` for financial domain  ✅ DRAFT

- **Goal:** Extraction prompt producing temporal-tagged triples from 10-K prose.
- **Status:** Draft delivered. Contains sections for fiscal-year resolution via chunk metadata, forward-looking guidance (`temporal_type="forward_looking"`), period-over-period comparison, and "what NOT to extract" (boilerplate + hypothetical risk prose). Wikipedia version archived at `prompts/archive/extract_v1_wikipedia.txt`.
- **Blocker:** 🔴 **Awaiting teammate review in `docs/prompt_review.md` — nobody assigned yet.**
- **Files:** `prompts/extract_v1.txt`, `prompts/archive/extract_v1_wikipedia.txt`, `docs/prompt_review.md`

#### A3a — 10-K acquisition + section parser (Mag5)  ✅ DELIVERED

- **Goal:** Download 25 filings; extract Items 1, 1A, 7, 7A, 8 as plain text.
- **Status:** Done. 25/25 filings downloaded + parsed, 0 silent skips, all 5 target sections present in every filing. See `docs/a3_report.md`.
- **Files:** `scripts/download_10k.py`, `src/parse_10k.py`, `tests/test_parse_10k.py`, `data/10k/raw/**`, `data/10k/sections/**`, `data/10k/manifest.json`

#### A3b — 10-K acquisition extension (new 5 tickers + FY2024)  **[NEW 2026-04-19]**

- **Goal:** Extend the 10-K corpus to 10 tickers × FY2022-2024 for alignment with FinReflectKG-MultiHop. Reuses A3 machinery.
- **Scope:**
  - 5 new tickers × FY2022-2024: **CSCO, ORCL, INTC, NVDA, ADBE** = 15 filings
  - Mag5 × FY2024: AAPL, MSFT, GOOGL, AMZN, META = 5 filings
  - Mag5 × FY2019-2021 (10 filings) are kept but dead for MultiHop eval (paper only covers 2022-2024); useful for home-grown retro questions
- **Dependencies:** None (same script path + SEC EDGAR)
- **Acceptance:**
  - [ ] 20 new filings downloaded to `data/10k/raw/{ticker}/FY{YYYY}.html`
  - [ ] `data/10k/manifest.json` updated; 45 filings total (25 Mag5 FY2019-2023 + 20 new)
  - [ ] All 5 sections parsable (same gate as A3a — 0 silent skips)
  - [ ] `docs/a3b_report.md` — extension log
- **Effort:** ~3-4 hours (rate-limited to SEC 10 req/sec)
- **Files:** `scripts/download_10k.py` (add tickers), `data/10k/raw/**`, `data/10k/sections/**`

#### A4a — Filter FinReflectKG-MultiHop 555 → our 10-ticker set  ✅ DELIVERED 2026-04-19

- **Goal:** External-validity QA slice from the FinReflectKG-MultiHop paper, filtered to our ticker set.
- **Status:** Done. **79 Qs** kept (intra 35 / inter_year 38 / cross_company 6). See `docs/multihop_filter_report.md`. Schema preserves evidence chunks for the judge.
- **Files:** `scripts/filter_multihop_qa.py`, `data/qa/multihop_filtered.jsonl`, `docs/multihop_filter_report.md`, `data/multihop_qa/final_master_dataset.json` (raw 555)

#### A4b — Home-grown temporal QA set  **[PENDING]**

- **Goal:** ~50 hand-authored Qs filling the gaps MultiHop under-covers: (a) explicit forward-looking, (b) fiscal-vs-calendar disambiguation, (c) additional cross-company pairs beyond the 6 MultiHop gave us, (d) inter-year comparisons grounded in Mag5 FY2019-2021 chunks.
- **Dependencies:** A3a (for FY2019-2021 chunks), A3b (for FY2024 + new tickers)
- **Acceptance:**
  - [ ] `data/qa/home_grown.jsonl` — 50 Qs, schema-compatible with A4a output (same `scope` field vocabulary: intra / inter_year / cross_company)
  - [ ] Each record: `{"qid", "question", "answer", "scope", "hop_count", "tickers", "years", "source_chunks", "evidence", "source_dataset": "home_grown_v1"}`
  - [ ] ≥15 explicitly cross-company (on top of MultiHop's 6)
  - [ ] ≥15 explicitly inter-year (matching MultiHop's hardest bucket)
  - [ ] Spot check by 1 teammate on 10 random Qs
- **Effort:** ~1-1.5 working days (authoring is the bulk; distributed across 4 teammates)
- **Files:** `data/qa/home_grown.jsonl`, `docs/home_grown_authoring.md`

#### A5 — Download FinReflectKG HF subset (baseline KG arm)

- **Goal:** Obtain 10-ticker × 2022-2024 subset of FinReflectKG as a **baseline KG arm** for comparing explicit-extraction temporal vs default filing-year temporal.
- **Status:** ✅ **10-ticker download complete 2026-04-19 (74,979 triples, 76.6% `extraction_type=default`).** META FY2022 has 0 triples in the HF release — known coverage gap; home-grown (A4b) can backfill if needed.
- **Dependencies:** None (HF streaming, no API cost)
- **Acceptance:**
  - [x] `scripts/download_finreflectkg.py` with `TARGET_TICKERS` = 10 tech
  - [x] `data/finreflectkg/triples.jsonl` — **74,979 triples delivered** (actuals per-ticker in `docs/finreflectkg_subset_report.md`)
  - [x] Schema preserved: `entity, entity_type, relationship, target, target_type, start_date, end_date, extraction_type, chunk_text` (18 fields)
  - [ ] `docs/finreflectkg_subset_report.md` refreshed with 10-ticker breakdown
- **Effort:** ~40 min wall-clock (streaming filter across 17.5M rows)
- **Files:** `scripts/download_finreflectkg.py`, `data/finreflectkg/*.jsonl`, `docs/finreflectkg_subset_report.md`

#### A6 — Hallucination guard in kg_extract  ✅ DELIVERED 2026-04-19

- **Goal:** Reject LLM-extracted triples whose `evidence` field is not a substring of the source chunk (normalized whitespace).
- **Status:** Done. `_validate_triple(triple, chunk_text)` and `parse_response(raw, chunk_text)` now enforce the substring check. 3 new tests: `test_parse_response_rejects_fabricated_evidence`, `test_parse_response_accepts_evidence_with_reflowed_whitespace`, `test_parse_response_rejects_evidence_with_altered_number`. 92/92 tests passing.
- **Files:** `src/kg_extract.py`, `tests/test_kg_extract.py`

### Checkpoint 0

- [x] 25 Mag5 FY2019-2023 filings parsed, section manifest clean
- [ ] 20 new filings (5 new tickers + Mag5 FY2024) parsed clean [A3b]
- [x] 3,810 Mag5 FY2019-2023 chunks sampled deterministically
- [ ] Full corpus re-sampled to ~4,600 chunks across 45 filings [A1 re-run after A3b]
- [x] Financial extraction prompt drafted
- [x] Hallucination guard shipped
- [ ] **Prompt reviewer assigned + sign-off in `docs/prompt_review.md`**
- [x] A4a MultiHop filter delivered (79 Qs)
- [ ] A4b Home-grown QA set (~50 Qs)
- [ ] A5 FinReflectKG 10-ticker subset downloaded (`b43mjr50m` running)
- [ ] `pytest tests/` green after each addition

### Phase 1 — Pilot

#### P1 — Pilot extraction (20 chunks)

- **Goal:** Run Gemini Flash on 20 financial chunks; catch prompt/schema bugs before full cost.
- **Dependencies:** A2 reviewed, A6 shipped (both status above)
- **Acceptance:**
  - [x] `src/kg_extract.py` with `extract_triples(chunk, prompt, client, cache)` and hallucination guard
  - [ ] `scripts/run_pilot.py` processes 20 chunks across ≥3 tickers for coverage
  - [ ] `results/pilot/log.jsonl`: latency, cost, triple count, parse failures, **hallucination-guard rejects**, raw response
  - [ ] **Hard guard:** total cost ≤ $0.20 (projected ~$0.006 at Gemini Flash rates)
- **Effort:** ~2 hours (kg_extract is already done)
- **Files:** `scripts/run_pilot.py`, `results/pilot/*.jsonl`

#### P2 — Pilot report + GO/NO-GO

- **Goal:** Decide if the prompt + schema survive financial prose.
- **Dependencies:** P1
- **Acceptance:**
  - [ ] `docs/pilot_report.md` with avg triples/chunk, non-null validity rate, parse-fail rate, cost/chunk, projected full-run cost, 5 good + 5 problematic examples, explicit **GO or NO-GO**
  - [ ] If NO-GO: iterate A2 → P1 → P2 (prompt loop)
  - [ ] If GO: team sign-off
- **Effort:** ~2 hours
- **Files:** `docs/pilot_report.md`

### Phase 2 — KG build + Baseline + Gold annotation (parallel)

#### B1 — Full KG build

- **Goal:** Extract triples from all ~4,600 chunks (projected after A1 re-run on 45-filing corpus); build NetworkX graph.
- **Dependencies:** P2 (GO), A1 re-run complete
- **Acceptance:**
  - [ ] `src/kg_build.py` with resume-on-failure
  - [ ] `results/kg/graph.pkl` — NetworkX `MultiDiGraph`, edges carry `{valid_from, valid_to, confidence, source_chunk_id, source_filing, temporal_type}`
  - [ ] `results/kg/build_stats.json`: triple count, non-null-validity rate, hallucination-guard rejections per ticker, total cost, wall-clock
  - [ ] **Hard guard:** total cost ≤ **$2.65** (single-pass projection at ~$0.57/k-chunk × 4,600 + 10% re-run buffer)
- **Effort:** 1 working day mostly wall-clock (cache-cold first pass; reruns are free)
- **Files:** `src/kg_build.py`, `results/kg/*`

#### B2 — FinReflectKG baseline KG  **[REPURPOSED]**

- **Goal:** Build a parallel KG from the downloaded FinReflectKG subset, on the same 10-ticker scope — so we can compare explicit-temporal (ours) against filing-year-default (theirs) under identical retrieval.
- **Dependencies:** A5 (10-ticker subset downloaded)
- **Acceptance:**
  - [ ] `src/baselines/finreflectkg_to_graph.py` — convert FinReflectKG triples into the same NetworkX schema as B1 (mapping `start_date`/`end_date` strings → `valid_from`/`valid_to` ints where parseable; `None` otherwise)
  - [ ] `results/kg/finreflectkg_graph.pkl`
  - [ ] `results/kg/finreflectkg_stats.json`: triple count, % with parseable dates vs nulls, % default vs explicit
  - [ ] Sanity: same 10 tickers, FY2022-2024 overlap ≥80% of years present
- **Effort:** ~1 day
- **Files:** `src/baselines/finreflectkg_to_graph.py`, `results/kg/finreflectkg_*`

#### T3.2 — Upgrade `src/eval.py` — LLM-Judge + BERTScore  **[NEW]**

- **Goal:** Replace F1/EM with paragraph-level metrics suitable for 10-K answers.
- **Dependencies:** None (pure library work)
- **Acceptance:**
  - [ ] `src/eval.py::score_with_judge(question, pred, gold, judge_client) -> float` — Gemini Flash judge prompt returns integer 0-10 (matching FinReflectKG-MultiHop for comparability); cached
  - [ ] `src/eval.py::score_bertscore(preds, golds) -> list[float]` — wraps `bert-score` library (F1 variant)
  - [ ] Stratified aggregation helper: `aggregate_by_scope(results, scope_field) -> dict`
  - [ ] Bootstrap CI helper preserved from v1
  - [ ] Tests with mocked judge (never hit real API in pytest)
- **Effort:** ~4 hours
- **Files:** `src/eval.py`, `tests/test_eval.py`, `prompts/judge_v1.txt`

#### B3 — Gold annotation (50 passages)

- **Goal:** 50-passage gold set for RQ3 extraction eval. Smaller than v1's 100 because budget-timeline.
- **Dependencies:** A3
- **Acceptance:**
  - [ ] `docs/annotation_protocol.md` — 5 worked examples from 10-K prose
  - [ ] 50 chunks annotated independently by 2 annotators
  - [ ] `src/iaa.py` run → **α reported without threshold** (per TA feedback #6)
  - [ ] Disagreements adjudicated; final gold at `data/annotations/rq3_gold_v2.jsonl`
  - [ ] `docs/annotation_iaa.md` reports α + adjudication counts (no pass/fail assertion)
- **Effort:** ~1 week wall-clock across 2 annotators
- **Files:** `data/annotations/rq3_gold_v2.jsonl`, `docs/annotation_*.md`

### Phase 3 — Pipeline (4 retrieval arms for L1/L2/L3 ablation)

#### G1 — Edge-level temporal filter

- **Goal:** Keep KG edges whose validity interval covers query year. **Default tolerance=0** (TA feedback #1: drop ±1 proof; tolerance is a knob, not a claim).
- **Dependencies:** B1
- **Acceptance:**
  - [ ] `src/pipeline/temporal_filter.py` — `filter_edges(edges, query_year, tolerance=0)`
  - [ ] Null-validity retained (conservative)
  - [ ] Interval-overlap semantics (valid_from ≤ q_year ≤ valid_to), handling `None` endpoints
  - [ ] Unit tests: vf=null, vt=null, both null, inside/outside window, forward-looking
- **Effort:** ~3 hours
- **Files:** `src/pipeline/temporal_filter.py`, `tests/test_temporal_filter.py`

#### G2 — Vanilla RAG arm

- **Goal:** Baseline BM25/dense retrieval over raw 10-K chunks, no KG, no temporal.
- **Dependencies:** A3, T3.2
- **Acceptance:**
  - [ ] `src/pipeline/vanilla_rag.py` — `retrieve(question, k=5) -> list[chunk_id]`
  - [ ] Dense retriever: sentence-transformers `all-MiniLM-L6-v2` (free) over chunk text
  - [ ] Unit tests on 5-Q toy set
- **Effort:** ~4 hours
- **Files:** `src/pipeline/vanilla_rag.py`, `tests/test_vanilla_rag.py`

#### G3 — Temporal-Vanilla-RAG arm  **[L1 — NEW]**

- **Goal:** Vanilla RAG + chunk-metadata temporal filter (TA-RAG style). Proves L1: temporal filtering helps even without a KG.
- **Dependencies:** G2
- **Acceptance:**
  - [ ] `src/pipeline/temporal_vanilla_rag.py` — `retrieve(question, query_year, k=5) -> list[chunk_id]`
  - [ ] Filter applied at chunk level using `chunk.fy` metadata (from `data/samples/10k_chunks.jsonl`)
  - [ ] Interval-overlap semantics identical to G1 for fair L1 vs L3 comparison
  - [ ] Unit tests: chunks from wrong FY are filtered out
- **Effort:** ~2 hours (shares retriever with G2)
- **Files:** `src/pipeline/temporal_vanilla_rag.py`, `tests/test_temporal_vanilla_rag.py`

#### G4 — KG²RAG arm  **[L2]**

- **Goal:** KG-guided chunk expansion (port of KG²RAG's retrieval path to financial KG). No temporal filter — isolates KG contribution.
- **Dependencies:** B1
- **Acceptance:**
  - [ ] `src/pipeline/kg2rag.py` — seed node identification + 1-hop expansion + chunk union
  - [ ] Returns ranked `list[chunk_id]` same shape as G2
  - [ ] Unit test on toy graph
- **Effort:** ~6 hours (most subtle of the four arms)
- **Files:** `src/pipeline/kg2rag.py`, `tests/test_kg2rag.py`

#### G5 — TempoRAG-KG Full arm  **[L3]**

- **Goal:** G4 + G1 composed. Proves L3: KG + Temporal beats each alone.
- **Dependencies:** G1, G4
- **Acceptance:**
  - [ ] `src/pipeline/temporag_kg_full.py` — KG²RAG retrieval, temporal filter applied to retrieved edges before chunk union
  - [ ] Falls back to KG²RAG behavior when query_year is None
  - [ ] Unit test: compared with G4 on temporal question, at least 1 edge filtered out
- **Effort:** ~2 hours (composition)
- **Files:** `src/pipeline/temporag_kg_full.py`, `tests/test_temporag_kg_full.py`

#### G6 — End-to-end integration + answer generation

- **Goal:** One `answer(question, arm, generator)` entry point exercising all 4 arms × 2 generators; 10-Q smoke test.
- **Dependencies:** G2, G3, G4, G5
- **Acceptance:**
  - [ ] `src/pipeline/run.py` — `answer(question, arm, generator, query_year=None) -> {"answer", "context", "edges", "arm", "generator"}`
  - [ ] 10-Q smoke test: all 4 arms × 2 generators return non-empty answers
  - [ ] Manual spot-check on 3 temporal Qs: L3 retrieves strictly fewer-or-equal edges than L2
- **Effort:** ~4 hours
- **Files:** `src/pipeline/run.py`, `tests/test_pipeline_integration.py`

### Checkpoint 1

- [ ] All 4 arms (G2/G3/G4/G5) pass unit tests
- [ ] 10-Q smoke test green across 4 arms × 2 generators (8 cells)
- [ ] On 3 temporal Qs: L3 demonstrably filters ≥1 edge vs L2

### Phase 4 — Evaluation (parallel)

#### E1 — RQ4 primary sweep  **[4-arm × 2-generator × stratified]**

- **Goal:** Produce the money table: L1/L2/L3 lift per arm, and ΔJudge(8B) vs ΔJudge(4o-mini) for the capability-parity story.
- **Dependencies:** B1, B2, T3.2, G6, A4 (QA set)
- **Acceptance:**
  - [ ] `scripts/run_rq4.py` sweeps **4 arms** {Vanilla, Temporal-Vanilla (L1), KG²RAG (L2), Full (L3)} × **2 generators** {Groq LLaMA-3.1-8B, GPT-4o-mini} × **3 scopes** {intra-doc, inter-year, cross-company} = **24 cells**
  - [ ] Additional **FinReflectKG-KG arm** on the same 4-arm × 2-generator matrix (subset of questions that match FinReflectKG coverage) — isolates explicit-temporal lift over default-filing-year
  - [ ] Reports **LLM-Judge (0-10)** + **BERTScore F1** with bootstrap CI per cell
  - [ ] Cost-per-question per cell (RQ4 secondary artefact)
  - [ ] Layer-level lifts: `Δ_L1 = Temporal-Vanilla − Vanilla`, `Δ_L2 = KG²RAG − Vanilla`, `Δ_L3 = Full − max(L1, L2)`
  - [ ] Capability-parity test: does LLaMA-3.1-8B on Full arm ≥ GPT-4o-mini on Vanilla arm?
  - [ ] **Hard guard:** total cost ≤ $5
- **Effort:** 2 days
- **Files:** `scripts/run_rq4.py`, `results/runs/rq4.json`, `results/runs/rq4_finreflectkg_arm.json`

#### E2 — RQ3 extraction accuracy

- **Goal:** Precision/Recall/F1 on 50-passage gold, plus hallucination-guard rejection rate breakdown.
- **Dependencies:** B1, B3
- **Acceptance:**
  - [ ] `scripts/eval_rq3.py` with per-pattern breakdown (explicit fiscal year, forward-looking, relative)
  - [ ] `results/runs/rq3_extraction.json` + hallucination-guard reject rate
- **Effort:** ~6 hours

#### E3 — RQ1 failure taxonomy

- **Goal:** Classify 50 baseline errors into 5 types.
- **Dependencies:** E1
- **Acceptance:**
  - [ ] `scripts/rq1_failure_analysis.py` extracts error set (cells where Vanilla answer was judged ≤4)
  - [ ] 50 manually classified into {Stale, Conflict, Missing, Relative, Hop-failure}
  - [ ] `results/runs/rq1_failures.md` with distribution + examples
- **Effort:** ~1 day

#### E4 — Test scenarios (TA feedback #4)

- **Goal:** 10 concrete failure→fix case studies for the progress report.
- **Dependencies:** E1
- **Acceptance:**
  - [ ] `docs/test_scenarios.md` — 10 cases, each with: question, baseline answer (wrong), edges retrieved by each arm, filtered answer (correct), brief analysis
  - [ ] ≥3 cases illustrate L1 lift (temporal-only fix without KG)
  - [ ] ≥3 cases illustrate L3 lift that neither L1 nor L2 alone achieves
- **Effort:** ~4 hours

#### E5 — XBRL ground-truth validation  **[NEW]**

- **Goal:** Catch numeric hallucinations in extracted triples by cross-checking against SEC XBRL Company Facts API (us-gaap ontology).
- **Dependencies:** B1 (KG built)
- **Acceptance:**
  - [ ] `scripts/xbrl_validate.py` — for each numeric triple (predicate suggests a financial metric), query `data.sec.gov/api/xbrl/companyfacts/CIK<n>.json` and check match within tolerance (±1% for floats)
  - [ ] Uses `edgartools` library (free, no API cost)
  - [ ] `results/runs/xbrl_validation.json`: % match, % null (XBRL doesn't cover), % disagreement with examples
  - [ ] Use as extraction-quality signal in progress report, NOT as a filter during retrieval
- **Effort:** ~1 day
- **Files:** `scripts/xbrl_validate.py`, `results/runs/xbrl_validation.json`

### Checkpoint 2

- [ ] L1/L2/L3 lift numbers with bootstrap CI
- [ ] Capability-parity table (8B-Full vs 4o-Vanilla)
- [ ] FinReflectKG-arm comparison numbers
- [ ] XBRL validation report
- [ ] RQ3 extraction P/R + hallucination-guard reject rate
- [ ] RQ1 taxonomy populated
- [ ] 10 test scenarios documented

### Phase 5 — Progress Report

#### R1 — Progress report writeup

- **Goal:** 8-section report addressing all TA feedback.
- **Dependencies:** Phase 4 (or write ahead with pending tables)
- **Acceptance:**
  - [ ] `proposal/progress_report.tex`: Abstract, Intro, Response to TA Feedback (7 points), Updated Methodology (10-K pivot), Experimental Setup, Results (RQ4 primary, RQ1/RQ3 secondary), Test Scenarios, Limitations & Risks, Revised Timeline, References
  - [ ] Cost breakdown table included
  - [ ] Builds cleanly; PDF committed
- **Effort:** 3–4 days
- **Files:** `proposal/progress_report.tex`, `proposal/progress_report.pdf`

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MultiHop 555-Q subset too small after strict ticker filter | Low | Medium | 10-ticker expansion yields **79 Qs (intra=35, inter_year=38, cross_company=6)** — largely retired; A4b home-grown set still targets cross_company thinness |
| Extraction accuracy on forward-looking statements is low | Medium | Medium | Treat forward-looking as reportable subtype, not failure; hallucination guard catches evidence-less inventions |
| Hallucination guard rejects too much (>20% of triples) | Medium | Medium | Tune whitespace-normalization; if still too strict, relax to token-set overlap as second pass |
| Gemini Flash pricing changes before pilot | Low | Low | Re-check at P1; pivot to GPT-4o-mini if blown |
| SEC HTML parse breaks on some filings | Low | Low | A3 delivered 25/25 clean — risk largely retired |
| Team throughput < 4 person-weeks in the 4-week window | Medium | High | E5 XBRL + E3 failure taxonomy are degradable |
| RQ4 ΔJudge shows no capability-parity | Medium | High (story) | Story pivots to "L1 + L2 + L3 each add measurable lift" — ablation still holds |
| L1 (Temporal-Vanilla) ≈ L3 (Full) — KG adds nothing | Medium | High (story) | Critical result — honestly report; L3 value shifts to cross-company queries where KG shortcuts matter |
| FinReflectKG-arm comparison confounded by their default-filing-year-only temporal | Low | Medium | Limit FinReflectKG-arm to questions whose gold year matches filing year; document the restriction |
| Retrieval pipeline (G2-G5) is the largest single work item, ~1 person-week | High | High | Pair-program G4 (most subtle); G3/G5 are compositions, unlikely to surprise |

## 8. Parallelization

| Phase | Parallel tasks | Owner suggestion |
|---|---|---|
| Phase 0 remaining | A3b (10-K extension), A4b (home-grown QA), A5 re-run (in flight), K4 lit scan | A3b Supanut (SEC fetch); A4b teammate; A5 Supanut (bg job); K4 any teammate |
| Phase 2 | B1, B2, B3, T3.2 | B1 Supanut (cost guard); B2+T3.2 Supanut (both Python-only); B3 two annotators |
| Phase 3 | G1+G2 first (shared by all arms); then G3/G4/G5 parallel | G4 pair-program (subtle); G3/G5 split |
| Phase 4 | E1, E2, E3, E4, E5 | E1 Supanut (largest); E5 Supanut (XBRL); others split |

## 9. Conventions (unchanged from v1)

- Python 3.9+, venv at `./venv`
- Secrets in `.env`, never commit
- All paid API calls via `src.cache.Cache`
- Seed 42 everywhere
- Tests under `tests/`, run `pytest tests/`
- Branches `feat/<task-id>-<short-name>` (e.g. `feat/a1-sampling-10k`)
- Commits in imperative mood, reference task ID

## 10. How to resume this project

1. Read this file (`tasks/plan.md`).
2. Read `tasks/todo.md` for the active checklist.
3. Read `docs/10k_scoping.md` for data-layer context.
4. Check memory: `memory/project_prior_work_landscape.md` for the comparative map; `memory/project_justification_chain.md` for the L1/L2/L3 argument.
5. Check state: `git log --oneline -10` and `pytest tests/`.
6. Pick the next unchecked task in the current phase.
7. Create `feat/<task-id>-<name>` branch; implement; test; commit; merge.
8. Update `tasks/todo.md` checkbox when merged.

## Appendix A — Granularity sub-study (deferred)

The locked decision is `int year` for `valid_from`/`valid_to`. A follow-up empirical question — "would quarter-level (`YYYY-Qn`) or ISO 8601 + precision flag yield measurable lift?" — is deferred to future work, with a lightweight pilot design in case we revisit:

1. Count % of temporal mentions in the 3,810-chunk corpus that are quarter-or-finer granular (via regex on a 100-chunk sample).
2. If >10% quarter-finer, re-run extraction with ISO 8601 + precision flag on the same 100 chunks and measure triple count + judge score on 10 questions per granularity level.
3. Compare lift vs integration cost (schema migration touches graph edges, temporal filter, eval harness).

This is an appendix, not a blocker. Progress report can cite it as "acknowledged limitation + clear next step".

## Appendix B — Reusable external artefacts

| Artefact | Source | Role in v2 |
|---|---|---|
| FinReflectKG 17.5M triples | HF `domyn/FinReflectKG` | Baseline KG arm (A5 → B2) |
| FinReflectKG-MultiHop 555 Qs | HF (public subset, to be confirmed) | QA set foundation (A4) |
| FinanceBench 150 Qs | published dataset | Fallback if MultiHop subset too small |
| SEC XBRL Company Facts API | `data.sec.gov/api/xbrl/` (free) | Ground-truth numeric validation (E5) |
| KG²RAG repo | `nju-websoft/KG2RAG` | Reference for KG-guided retrieval implementation (G4) |
