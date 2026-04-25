# TempoRAG-KG v2 — Active Todo List

**See `tasks/plan.md` for full task details, acceptance criteria, and verification steps.**
**See `docs/10k_scoping.md` for the 10-K data scope.**

Current phase: **Phase 0 — Adaptation (pivot to 10-K)**
Progress report target: **2026-05-15** (~4 weeks out)

---

## v1 tasks (superseded — preserved in `tasks/archive/`)

Phase 1 infra (T1–T4) is **reused as-is**; T5 will be rewritten.

- [x] **T1** — Deterministic sampling for HotpotQA/MuSiQue (`src/sampling.py`) — reusable base; A1 extends it
- [x] **T2** — Cache layer (`src/cache.py`) — reused as-is
- [x] **T3** — Eval harness (`src/eval.py`) — reused as-is
- [x] **T4** — IAA script (`src/iaa.py`) — reused as-is (drop α≥0.70 assertion in annotation doc)
- [~] **T5** — Extraction prompt for Wikipedia — **superseded by A2** (rewrite for financial)

Other v1 tasks (T6–T18) superseded by v2 phases below. See `tasks/archive/plan_v1_hotpotqa.md` if needed.

---

## Pivot kickoff (K series, from `/Users/supanut.k/.claude/plans/let-s-recap-this-floating-clarke.md`)

- [x] **K5** — `docs/10k_scoping.md` (5 tickers × 5 yrs; cost projection ~$0.77)
- [x] **K1** — Overwrite `tasks/plan.md` with v2
- [~] **K3** — Update this file (in progress)
- [ ] **K2** — `docs/ta_feedback_for_team_2026_04_16.md` (English reference for team)
- [ ] **K4** — `docs/temporal_methods_scan.md` (3–5 papers — TA feedback #5)

---

## Prereqs from user

- [ ] Activate venv; `pip install -r requirements.txt` (new deps: `sec-edgar-downloader` or `requests`, `beautifulsoup4`)
- [ ] `.env` populated with `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, `RANDOM_SEED=42`

## Phase 0 — Adaptation (current)

- [ ] **A1** — Adapt `src/sampling.py` for 10-K chunk sampling
- [ ] **A2** — Rewrite `prompts/extract_v1.txt` for financial domain (archive Wikipedia version)
- [ ] **A3** — `scripts/download_10k.py` + `src/parse_10k.py` (25 filings, sections I/1A/7/7A/8)
- [ ] **A4** — Question set: FinanceBench subset + `scripts/synthesize_questions.py` (≥100 Q)

### Checkpoint 0 ⛔
- [ ] 25 filings parsed; manifest clean
- [ ] Prompt reviewed by ≥1 teammate in `docs/prompt_review.md`
- [ ] ≥100 QA pairs, ≥50 temporal
- [ ] `pytest tests/` green

## Phase 1 — Pilot

- [ ] **P1** — Pilot extraction on 20 chunks (hard $0.20 cost guard)
- [ ] **P2** — `docs/pilot_report.md` + **GO/NO-GO**

## Phase 2 — KG build + Baseline + Gold (parallel)

- [ ] **B1** — Full KG build (hard $2 cost guard)
- [ ] **B2** — KG²RAG baseline for financial QA
- [ ] **B3** — 50-passage gold annotation (2 annotators; α reported, no threshold)

## Phase 3 — Pipeline

- [ ] **G1** — Temporal filter (tolerance=0 default)
- [ ] **G2** — GEAR beam search
- [ ] **G3** — GoG fill-in *(optional; defer if tight)*
- [ ] **G4** — End-to-end integration test (10 Q)

### Checkpoint 1 ⛔
- [ ] Pipeline unit tests green
- [ ] 10-Q integration passes; filter affects ≥3 temporal Q

## Phase 4 — Evaluation (parallel)

- [ ] **E1** — **RQ4 primary sweep** (2 gen × 3 cond × temporal subset; hard $5 cap)
- [ ] **E2** — RQ3 extraction accuracy on 50-passage gold
- [ ] **E3** — RQ1 failure taxonomy on 50 baseline errors
- [ ] **E4** — `docs/test_scenarios.md` (10 failure→fix cases — TA feedback #4)

### Checkpoint 2 ⛔
- [ ] RQ4 table complete (bootstrap CIs)
- [ ] RQ3 & RQ1 results recorded
- [ ] 10 test scenarios documented

## Phase 5 — Progress Report

- [ ] **R1** — `proposal/progress_report.tex` (8 sections + appendix)

---

## Team assignment (fill in after K2 shared)

| Task | Owner | Notes |
|---|---|---|
| A1 | ? | extends existing sampling |
| A2 | ? | prompt engineering; need financial-literate reviewer |
| A3 | ? | SEC HTML parsing |
| A4 | ? | needs review of FinanceBench access |
| B3 annotator A | ? | |
| B3 annotator B | ? | different from A |
| K2 | Supanut | next |
| K4 | ? | lit scan — good for a teammate |

---

## Quick commands

```bash
# activate env
source venv/bin/activate

# run all tests
pytest tests/ -v

# see current progress
git log --oneline -10
cat tasks/todo.md
```
