# TempoRAG-KG — Implementation Plan

**Last updated:** 2026-04-16
**Status:** Phase 0 (bootstrap) complete. Starting Phase 1 (infrastructure).
**Repo:** https://github.com/gossbu666/TempoRAG-KG
**Related docs:** `proposal/temporag_kg_report_rev2.tex`, `tasks/todo.md`, `README.md`

---

## 1. Purpose of this document

This is the **single source of truth** for what we are building, in what order, and how each piece is verified. It is designed to be read by any team member or by a Claude session in VSCode and used to continue work without re-deriving context.

If this document and memory disagree, **this document wins**. Update it when plans change; do not rely on conversation history.

---

## 2. Current state snapshot

| Category | Status |
|---|---|
| Literature review (16 papers) | ✅ Done |
| EDA (HotpotQA 35%, MuSiQue 38.3% temporal) | ✅ Done |
| Proposal submitted (March 23) | ✅ Done |
| TA feedback received (3 methodological items) | ✅ Documented |
| April revision plan | ✅ Documented (memory + this file) |
| GitHub repo scaffolding | ✅ Pushed to `main` |
| Python venv (3.9) + requirements | ✅ Set up locally |
| `.env` with real API keys | ⏳ User to create |
| Phase 1 — Infrastructure | ⏳ Next |
| Phase 2 — Pilot de-risking | ⏳ After Phase 1 |
| Phase 3–7 | ⏳ Sequential |

---

## 3. Budget and operating constraints

- **Hard cap:** $20 total across all LLM API spend
- **Cache-first policy:** Every paid API call must go through `src.cache.Cache`
- **Free tier priority:** Groq LLaMA-3.1-8B is the primary generator (free)
- **Pilot gate:** No full-scale extraction until 20-chunk pilot passes go/no-go
- **Seed:** `RANDOM_SEED=42` everywhere (set in `.env`)
- **Determinism:** Sampling, train/test splits, bootstrap samples all seeded

---

## 4. Dependency graph

```
Phase 0 — Bootstrap (DONE)
    │
    ▼
Phase 1 — Infrastructure (parallelizable)
    ├─ T1 Sampling
    ├─ T2 Cache layer
    ├─ T3 Eval harness (F1/EM/bootstrap)
    ├─ T4 IAA script (Krippendorff α)
    └─ T5 Extraction prompt v1
                    │
                    ▼
            ╔══════════════════╗
            ║  CHECKPOINT 1    ║  All unit tests green; prompt reviewed
            ╚══════════════════╝
                    │
                    ▼
Phase 2 — Pilot de-risking
    ├─ T6 Pilot extraction (20 chunks, Gemini Flash)
    └─ T7 Pilot report + GO/NO-GO decision
                    │
                    ▼
Phase 3 — Baseline + KG + Annotation (parallelizable after T7)
    ├─ T8 Full KG build
    ├─ T9 KG²RAG baseline reproduction
    └─ T10 RQ3 annotation (100 passages, 2 annotators)
                    │
                    ▼
Phase 4 — Pipeline
    ├─ T11 Temporal filter (±1 yr tolerance)
    ├─ T12a GEAR beam search
    ├─ T12b GoG fill-in
    └─ T12c End-to-end integration test
                    │
                    ▼
            ╔══════════════════╗
            ║  CHECKPOINT 2    ║  Pipeline returns answers on 10 Q end-to-end
            ╚══════════════════╝
                    │
                    ▼
Phase 5 — Evaluation (parallelizable)
    ├─ T13 RQ3 extraction eval
    ├─ T14 RQ1 failure mode analysis
    ├─ T15 RQ2 ablation (3 conditions × 2 datasets)
    └─ T16 RQ4 generator ablation
                    │
                    ▼
            ╔══════════════════╗
            ║  CHECKPOINT 3    ║  All 4 RQs have numbers + bootstrap CIs
            ╚══════════════════╝
                    │
                    ▼
Phase 6 — Progress Report
    └─ T17 Write progress_report.tex (8 sections)
                    │
                    ▼
Phase 7 — Final Report
    └─ T18 Final LaTeX + slides + optional demo
```

---

## 5. Task details

Each task is a **vertical slice** — independently verifiable by running a test, producing a file, or completing a review. Tasks that should run in parallel are tagged.

### Phase 1 — Infrastructure

#### T1 — Deterministic sampling

- **Goal:** Produce reproducible sampled question sets used by every downstream run.
- **Dependencies:** None (raw data in `data/`, EDA in `results/temporal_eda_results.json`)
- **Parallelizable with:** T2, T3, T4, T5
- **Acceptance criteria:**
  - [ ] `src/sampling.py` exposes `sample_hotpot()` and `sample_musique()` functions
  - [ ] Reads `RANDOM_SEED` from `.env` (default 42)
  - [ ] Emits `data/samples/hotpot_1000.json` and `data/samples/musique_500.json`
  - [ ] Each record: `{"id": str, "question": str, "temporal": bool, "hop_count": int|null, "patterns": list[str]}`
  - [ ] HotpotQA sample = exactly 500 temporal + 500 non-temporal
  - [ ] MuSiQue sample = 500 stratified across 2/3/4 hops proportional to full dev set
  - [ ] Re-running is byte-identical (SHA256 stable)
- **Verification:**
  - `pytest tests/test_sampling.py -v` → all pass
  - `sha256sum data/samples/*.json` identical across two independent runs
- **Artifacts:** `src/sampling.py`, `tests/test_sampling.py`, `data/samples/hotpot_1000.json`, `data/samples/musique_500.json`
- **Effort:** ~2 hours

#### T2 — Cache layer

- **Goal:** Disk-backed JSON cache wrapping every paid API call. Zero cost on repeats.
- **Dependencies:** None
- **Parallelizable with:** T1, T3, T4, T5
- **Acceptance criteria:**
  - [ ] `src/cache.py` provides class `Cache(cache_dir: Path)`
  - [ ] Methods: `get(key) -> dict | None`, `put(key, value) -> None`, `key_for(model, prompt, params) -> str`
  - [ ] Key = `sha256(f"{model}|{prompt}|{json.dumps(params, sort_keys=True)}")`
  - [ ] Files stored under `cache/{key[:2]}/{key[2:]}.json` (avoids one-giant-dir)
  - [ ] Thread-safe reads; writes are last-writer-wins (acceptable)
  - [ ] Never raises on cache miss — returns `None`
- **Verification:**
  - `pytest tests/test_cache.py -v` covers: hit, miss, put→get roundtrip, different params produce different keys, persistence across `Cache` instance recreation
- **Artifacts:** `src/cache.py`, `tests/test_cache.py`
- **Effort:** ~2 hours

#### T3 — Eval harness

- **Goal:** Token-level F1, Exact Match (EM), and bootstrap CI.
- **Dependencies:** None
- **Parallelizable with:** T1, T2, T4, T5
- **Acceptance criteria:**
  - [ ] `src/eval.py` provides: `f1_token(pred, gold)`, `em(pred, gold)`, `bootstrap_ci(scores, n=1000, alpha=0.05)`
  - [ ] Normalization: lowercase, strip articles (a/an/the), strip punctuation, collapse whitespace (SQuAD convention)
  - [ ] `aggregate(preds, golds)` returns `{"f1_mean": float, "f1_ci": (lo, hi), "em_mean": float, "em_ci": (lo, hi), "n": int}`
  - [ ] Handles multi-answer gold sets (take max F1 over alternatives)
- **Verification:**
  - `pytest tests/test_eval.py -v`
  - Spot check: reproduce 3 known-answer F1 scores from HotpotQA official scorer within 0.001
- **Artifacts:** `src/eval.py`, `tests/test_eval.py`
- **Effort:** ~2 hours

#### T4 — IAA script (Krippendorff's α, interval)

- **Goal:** Compute Krippendorff's α with interval distance for year annotations (required for RQ3 response to TA feedback).
- **Dependencies:** `krippendorff` package in `requirements.txt`
- **Parallelizable with:** T1, T2, T3, T5
- **Acceptance criteria:**
  - [ ] `src/iaa.py` exposes `krippendorff_alpha_interval(ratings: list[list[int|None]]) -> float`
  - [ ] Accepts missing values (None / np.nan)
  - [ ] CLI: `python -m src.iaa annotations.csv` prints α
  - [ ] Unit test matches Krippendorff (2004) canonical example within 0.005
- **Verification:**
  - `pytest tests/test_iaa.py -v`
- **Artifacts:** `src/iaa.py`, `tests/test_iaa.py`
- **Effort:** ~1 hour

#### T5 — Extraction prompt v1

- **Goal:** Finalize Gemini Flash extraction prompt. Handle (a) multi-tenure entities and (b) relative/ambiguous temporal references.
- **Dependencies:** None
- **Parallelizable with:** T1, T2, T3, T4
- **Acceptance criteria:**
  - [ ] `prompts/extract_v1.txt` committed
  - [ ] Explicit instruction for **multi-tenure** ("output one triple per continuous period")
  - [ ] Explicit instruction for **relative refs** ("set both `valid_from` and `valid_to` to null; set `metadata.temporal_type` to `relative`")
  - [ ] JSON schema documented in prompt
  - [ ] 3 worked examples: (i) explicit year, (ii) multi-tenure, (iii) relative reference
  - [ ] Reviewed by one teammate (comment committed to `docs/prompt_review.md` or PR)
- **Verification:**
  - Teammate review recorded
- **Artifacts:** `prompts/extract_v1.txt`, `docs/prompt_review.md`
- **Effort:** ~1 hour

### Checkpoint 1

**Gate to Phase 2:**
- [ ] All of T1–T5 marked done
- [ ] `pytest` on full suite passes (0 failures)
- [ ] Prompt reviewed by at least one non-author
- [ ] `.env` populated with real `GEMINI_API_KEY` and `GROQ_API_KEY`

### Phase 2 — Pilot de-risking

#### T6 — Pilot extraction (20 chunks)

- **Goal:** Run Gemini 1.5 Flash on 20 HotpotQA chunks through the cache; capture operational metrics before scaling.
- **Dependencies:** T1 (for chunk source), T2 (cache), T5 (prompt)
- **Acceptance criteria:**
  - [ ] `src/kg_extract.py` exposes `extract_triples(chunk_text, prompt, cache) -> list[dict]`
  - [ ] `scripts/run_pilot.py` processes 20 chunks and writes `results/pilot/log.jsonl` + `results/pilot/raw.jsonl`
  - [ ] Log records: latency, estimated cost, parsed triple count, parse failures, raw response
  - [ ] **Hard guard:** total cost ≤ $0.50 — abort if exceeded
- **Verification:**
  - `python scripts/run_pilot.py` completes with exit 0
  - `jq '. | length' results/pilot/log.jsonl` returns 20
- **Artifacts:** `src/kg_extract.py`, `scripts/run_pilot.py`, `results/pilot/log.jsonl`, `results/pilot/raw.jsonl`
- **Effort:** ~3 hours

#### T7 — Pilot report + GO/NO-GO

- **Goal:** Decide whether the extraction prompt is good enough for the full KG build.
- **Dependencies:** T6
- **Acceptance criteria:**
  - [ ] `docs/pilot_report.md` with: avg triples/chunk, non-null validity rate, parse-fail rate, cost/chunk, projected full-run cost, 5 good + 5 problematic examples, explicit **GO or NO-GO** decision
  - [ ] If NO-GO: revise prompt → re-pilot (loop T5 → T6 → T7)
  - [ ] If GO: team sign-off logged
- **Verification:**
  - Report exists with decision section
  - Team member acknowledgment in git log or PR comment
- **Artifacts:** `docs/pilot_report.md`
- **Effort:** ~2 hours

### Phase 3 — Baseline + KG + Annotation (parallel)

#### T8 — Full KG build

- **Goal:** Extract triples from all chunks in sampled HotpotQA + MuSiQue; build NetworkX graph; serialize.
- **Dependencies:** T7 (go decision)
- **Parallelizable with:** T9, T10
- **Acceptance criteria:**
  - [ ] `src/kg_build.py` orchestration with resume-on-failure support
  - [ ] Chunking: 512 tokens, 100-token overlap (per KG²RAG)
  - [ ] `results/kg/graph.pkl` — NetworkX `MultiDiGraph`, edges carry `{valid_from, valid_to, confidence, source_chunk_id, source_doc_id}`
  - [ ] `results/kg/build_stats.json` with: total triples, non-null-validity rate, total cost, wall-clock
  - [ ] **Hard guard:** total cost ≤ $5 — abort if exceeded
- **Verification:**
  - Load graph in Python REPL; sample 10 random edges; spot-check correctness
- **Artifacts:** `src/kg_build.py`, `results/kg/graph.pkl`, `results/kg/build_stats.json`
- **Effort:** 1 working day (mostly wall-clock)

#### T9 — KG²RAG baseline reproduction

- **Goal:** Reproduce KG²RAG on sampled HotpotQA with Groq LLaMA-3.1-8B; confirm faithful reimplementation.
- **Dependencies:** T8, T3
- **Parallelizable with:** T8 (after KG ready), T10
- **Acceptance criteria:**
  - [ ] `src/baselines/kg2rag.py` implements KG-guided chunk expansion + MST context organization per Zhu et al. 2025
  - [ ] Run on 1000 sampled HotpotQA questions
  - [ ] F1 overall within 2.0 points of paper number (Table 3 reports ~85.6)
  - [ ] `results/runs/kg2rag_baseline.json` with bootstrap CI
- **Verification:**
  - Reproduction delta ≤ 2.0 F1 points documented in results file
- **Artifacts:** `src/baselines/kg2rag.py`, `results/runs/kg2rag_baseline.json`
- **Effort:** 2 days

#### T10 — RQ3 annotation

- **Goal:** Hand-annotate 100 HotpotQA passages with ground-truth `[valid_from, valid_to]`.
- **Dependencies:** T1 (passages sampled), T4 (α script)
- **Parallelizable with:** T8, T9 (different humans)
- **Acceptance criteria:**
  - [ ] `docs/annotation_protocol.md` with 5–10 worked examples (explicit year, multi-tenure, relative, conflict, missing)
  - [ ] 100 passages annotated independently by 2 annotators
  - [ ] `src/iaa.py` run → α ≥ 0.70 target (moderate-to-strong)
  - [ ] Disagreements adjudicated; final gold at `data/annotations/rq3_gold_v1.jsonl`
  - [ ] `docs/annotation_iaa.md` reports α + adjudication counts
- **Verification:**
  - α ≥ 0.70 documented
- **Artifacts:** `docs/annotation_protocol.md`, `docs/annotation_iaa.md`, `data/annotations/rq3_gold_v1.jsonl`
- **Effort:** ~1 working week wall-clock, 10–15 hr per annotator

### Phase 4 — Pipeline

#### T11 — Temporal filter

- **Goal:** Keep edges whose validity interval covers the query year, with ±1 year tolerance.
- **Dependencies:** T8
- **Acceptance criteria:**
  - [ ] `src/pipeline/temporal_filter.py` exposes `filter_edges(edges, query_year, tolerance=1) -> list`
  - [ ] Equation: `(vf is None or vf ≤ query_year + τ) AND (vt is None or vt ≥ query_year − τ)`
  - [ ] Null-validity edges retained (conservative); counted separately in a returned stats dict
  - [ ] Unit tests cover: `vf=null`, `vt=null`, both null, both set (inside/outside window, on boundary)
- **Verification:**
  - `pytest tests/test_temporal_filter.py -v`
- **Artifacts:** `src/pipeline/temporal_filter.py`, `tests/test_temporal_filter.py`
- **Effort:** ~3 hours

#### T12a — GEAR beam search

- **Goal:** Beam-search graph traversal scoring neighbors by cosine similarity to subgoal.
- **Dependencies:** T8
- **Acceptance criteria:**
  - [ ] `src/pipeline/gear.py` exposes `beam_search(graph, start_nodes, subgoal_emb, beam=3, depth=2)`
  - [ ] Deterministic tie-breaking (stable ordering)
  - [ ] Unit test on 5-node toy graph
- **Verification:**
  - `pytest tests/test_gear.py -v`
- **Artifacts:** `src/pipeline/gear.py`, `tests/test_gear.py`
- **Effort:** ~4 hours

#### T12b — GoG fill-in

- **Goal:** LLM fill-in on dead-end beams; retain facts with confidence ≥ 0.7.
- **Dependencies:** T2 (cache), T12a
- **Acceptance criteria:**
  - [ ] `src/pipeline/gog.py` exposes `fill_in(query, partial_graph, confidence_threshold=0.7)`
  - [ ] Uses primary generator (Groq LLaMA-3.1-8B) via cache
  - [ ] Unit test with mocked LLM response
- **Verification:**
  - `pytest tests/test_gog.py -v`
- **Artifacts:** `src/pipeline/gog.py`, `tests/test_gog.py`
- **Effort:** ~3 hours

#### T12c — End-to-end integration

- **Goal:** Prove the pipeline end-to-end on real questions.
- **Dependencies:** T11, T12a, T12b
- **Acceptance criteria:**
  - [ ] `src/pipeline/run.py` exposes `answer(question: str) -> dict` with keys `answer`, `context`, `filtered_edges`, `gog_fills`
  - [ ] Integration test on 10 HotpotQA questions — each returns non-empty answer
  - [ ] Latency < 10s/query (soft target)
- **Verification:**
  - `pytest tests/test_pipeline_integration.py -v`
- **Artifacts:** `src/pipeline/run.py`, `tests/test_pipeline_integration.py`
- **Effort:** ~4 hours

### Checkpoint 2

**Gate to Phase 5:**
- [ ] All pipeline unit tests green
- [ ] Integration test: 10 Q → 10 answers returned
- [ ] Manual spot-check on 3 temporal questions shows filter affecting results
- [ ] `results/kg/build_stats.json` non-null validity rate ≥ 30% (otherwise filter is mostly no-op — flag as limitation)

### Phase 5 — Evaluation

#### T13 — RQ3 extraction eval

- **Goal:** Evaluate extraction accuracy vs. 100-passage gold set.
- **Dependencies:** T3, T8, T10
- **Acceptance criteria:**
  - [ ] `scripts/eval_rq3.py` computes precision/recall/F1 with ±1 yr tolerance
  - [ ] Breakdown by pattern: explicit year, implicit/relative (Type 3b), conflicting
  - [ ] `results/runs/rq3_extraction.json` populated
- **Effort:** 1 day

#### T14 — RQ1 failure mode analysis

- **Goal:** Classify 100 KG²RAG errors into 5 types (1, 2, 3, 3b, 4).
- **Dependencies:** T9
- **Acceptance criteria:**
  - [ ] `scripts/rq1_failure_analysis.py` extracts error set
  - [ ] 100 errors manually classified
  - [ ] `results/runs/rq1_failures.md` with distribution + examples
- **Effort:** 1 day

#### T15 — RQ2 ablation

- **Goal:** 3-condition ablation (Vanilla / KG²RAG / Full) × 2 datasets.
- **Dependencies:** T9, T12c
- **Acceptance criteria:**
  - [ ] `scripts/run_ablation.py` runs all 6 condition × dataset combinations
  - [ ] F1/EM with CIs on overall + temporal + non-temporal subsets
  - [ ] `results/runs/rq2_ablation.json`
  - [ ] **Hard guard:** total cost ≤ $2
- **Effort:** 2 days

#### T16 — RQ4 generator ablation

- **Goal:** Full TempoRAG-KG with 4 generators × temporal subset.
- **Dependencies:** T15
- **Acceptance criteria:**
  - [ ] `scripts/run_rq4.py` sweeps LLaMA-8B / LLaMA-70B / Flash / 4o-mini
  - [ ] ΔF1 = (Full − KG²RAG) computed per generator
  - [ ] Inverse-scaling hypothesis tested
  - [ ] **Hard guard:** cost ≤ $3
- **Effort:** 2 days

### Checkpoint 3

**Gate to Phase 6:**
- [ ] RQ1–RQ4 all have numerical results with bootstrap CIs
- [ ] Failure mode examples collected for writeup
- [ ] Null-retention coverage documented

### Phase 6 — Progress Report

#### T17 — Progress Report writeup

- **Goal:** 8-section report per agreed structure.
- **Dependencies:** Either all of Phase 5, OR write ahead with "pending" tables + revised methodology
- **Acceptance criteria:**
  - [ ] `proposal/progress_report.tex` with sections: Abstract, Introduction, Response to TA Feedback, Revised Scope, New RQ4, Updated Methodology, Progress to Date, Risks & Mitigations, Revised Timeline, References, Appendix
  - [ ] All 3 TA feedback items explicitly addressed
  - [ ] Cost breakdown table included
  - [ ] PDF builds cleanly
- **Effort:** 2–3 days

### Phase 7 — Final deliverables

#### T18 — Final artifacts

- **Goal:** Final report + slides + optional Streamlit demo.
- **Acceptance criteria:**
  - [ ] `proposal/final_report.tex` with full results
  - [ ] `slides/final_presentation.pdf`
  - [ ] Repo `README.md` updated with final status
  - [ ] (Optional) Streamlit demo if time remains
- **Effort:** 3–5 days

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Extraction quality on implicit/relative refs low | Medium | High | Conservative null-retention; report coverage explicitly |
| HotpotQA not a true temporal benchmark | High (known) | Medium | Acknowledge in limitations; TimeQuestions as stretch |
| Budget overrun on extraction | Low (pilot-gated) | High | Hard cost guards in every script; cache-first |
| Annotation α < 0.70 | Medium | Medium | Expand protocol with more examples; adjudicate rounds |
| KG²RAG reproduction drifts > 2 F1 | Medium | Medium | Document delta; ensure fair comparison despite gap |
| Multi-tenure entities break schema | Low | Medium | Prompt explicitly handles; pilot validates |
| Teammate workload imbalance | Medium | Medium | Task assignments in `tasks/todo.md`; weekly sync |

---

## 7. Conventions

- **Python:** 3.9+, venv at `./venv`
- **Secrets:** `.env` only; never commit
- **API calls:** always via `src.cache.Cache` wrapper
- **Seed:** 42 (in `.env` as `RANDOM_SEED`)
- **Tests:** under `tests/`, run `pytest tests/` (targeting ≥ 80% line coverage for `src/` eventually; not strict during early phases)
- **Branches:** `feat/<task-id>-<short-name>` (e.g. `feat/t1-sampling`) — one PR per task, self-merge acceptable
- **Commits:** imperative mood, reference task ID (e.g., `T2: add disk-backed JSON cache`)
- **Results files:** JSON under `results/runs/`, markdown reports under `docs/` or `results/runs/`

---

## 8. Parallelization summary

| Can run in parallel | Tasks |
|---|---|
| Phase 1 fully parallel | T1, T2, T3, T4, T5 |
| Phase 3 fully parallel (after T7) | T8, T9, T10 |
| Phase 5 fully parallel (after T12c) | T13, T14, T15, T16 |

Phase 2 and Phase 4 have internal sequential dependencies.

---

## 9. How to resume this project

1. Read this file (`tasks/plan.md`).
2. Read `tasks/todo.md` for the current active checklist.
3. Check latest state: `git log --oneline -10` and `pytest tests/`.
4. Pick the next unchecked task in the current phase.
5. Create branch `feat/<task-id>-<name>`; implement; test; PR.
6. Update `tasks/todo.md` checkbox when merged.
