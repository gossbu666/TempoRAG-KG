# TempoRAG-KG — Active Todo List

**See `tasks/plan.md` for full task details, acceptance criteria, and verification steps.**

Current phase: **Phase 1 — Infrastructure**

---

## Phase 0 — Bootstrap ✅

- [x] GitHub repo created and pushed
- [x] Directory structure scaffolded
- [x] `.gitignore`, `.env.example`, `requirements.txt`, `README.md`

## Phase 1 — Infrastructure (in progress)

- [ ] **T1** — Deterministic sampling (`src/sampling.py` + tests + `data/samples/*.json`)
- [x] **T2** — Cache layer (`src/cache.py` + tests)
- [ ] **T3** — Eval harness (`src/eval.py` F1/EM/bootstrap + tests)
- [ ] **T4** — IAA script (`src/iaa.py` Krippendorff α + tests)
- [ ] **T5** — Extraction prompt v1 (`prompts/extract_v1.txt` + teammate review)

### Prereqs from user
- [ ] Create `.env` from `.env.example` with real `GEMINI_API_KEY`, `GROQ_API_KEY`
- [ ] Activate venv, run `pip install -r requirements.txt`

### Checkpoint 1 ⛔
- [ ] All of T1–T5 done
- [ ] `pytest tests/` all green
- [ ] Prompt reviewed by ≥1 teammate
- [ ] `.env` populated

## Phase 2 — Pilot de-risking

- [ ] **T6** — Pilot extraction (20 chunks, Gemini Flash, hard $0.50 cost guard)
- [ ] **T7** — Pilot report + **GO/NO-GO decision**

## Phase 3 — Baseline + KG + Annotation (parallel)

- [ ] **T8** — Full KG build (hard $5 cost guard)
- [ ] **T9** — KG²RAG baseline reproduction (Groq LLaMA-8B, free)
- [ ] **T10** — RQ3 annotation (100 passages × 2 annotators, α ≥ 0.70)

## Phase 4 — Pipeline

- [ ] **T11** — Temporal filter with ±1 year tolerance
- [ ] **T12a** — GEAR beam search
- [ ] **T12b** — GoG fill-in
- [ ] **T12c** — End-to-end integration (10-question smoke test)

### Checkpoint 2 ⛔
- [ ] Pipeline returns answers on 10 Q
- [ ] Null-validity coverage documented
- [ ] 3 temporal questions show filter affecting results

## Phase 5 — Evaluation

- [ ] **T13** — RQ3 extraction eval (precision/recall/F1 × 3 pattern types)
- [ ] **T14** — RQ1 failure mode analysis (100 errors classified into 5 types)
- [ ] **T15** — RQ2 ablation (3 conditions × 2 datasets, hard $2 cost guard)
- [ ] **T16** — RQ4 generator ablation (4 generators × temporal subset, hard $3 cost guard)

### Checkpoint 3 ⛔
- [ ] All 4 RQs have numbers + bootstrap CIs

## Phase 6 — Progress Report

- [ ] **T17** — `proposal/progress_report.tex` (8 sections + appendix)

## Phase 7 — Final deliverables

- [ ] **T18** — Final report + slides + optional demo

---

## Team assignment (to fill in once teammates onboard)

| Task | Owner |
|---|---|
| T1 | ? |
| T2 | ? |
| T3 | ? |
| T4 | ? |
| T5 | ? |
| T10 annotator A | ? |
| T10 annotator B | ? |

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
