# Sub-project (c) — Failure Taxonomy for TempoRAG-KG

**Date:** 2026-04-25
**Feeds:** `proposal/final_report.tex` §6 Discussion + §6.1 Limitations + Appendix D,
`docs/final_slides.pptx` Slide 7.
**Status:** Design approved. Ready for implementation plan.

## 1. Context & Motivation

Presentation feedback on 2026-04-20 included one team-specific ask
(TA comment #2):

> "Does it have any categories that the model cannot answer or not?
>  Even like this, based on the business document, 10-K, 10-Q, does it
>  have any answer it cannot answer?"

This is two questions in one:

- **(A) Model-level failures** — the corpus has the information but the
  model still fails (retrieval miss, hallucination, style artefact, etc.).
- **(B) Corpus-level limits** — the question is structurally
  unanswerable from 10-K / 10-Q content (stock prices on specific dates,
  forecasts, opinions, forward-looking claims).

The final report's Discussion section (rubric 3 pts) is the
right home for this analysis. Additionally, the general-feedback items
#4 "qualitative analysis" and #5 "distribution of errors" both land
here: one worked example per category + per-cell counts.

This spec defines a **reusable, reproducible classifier** — not a
one-shot manual analysis — so the same tool runs on future eval sweeps
(including the expanded QA set after sub-(d) vet completes).

## 2. Goals

1. Assign every `(model, condition, question)` prediction a **primary
   failure category** (or the non-failure label if the prediction is
   correct or the model correctly abstained).
2. Optionally assign a **secondary cause** for compound failures.
3. Produce four count tables: `model × category`, `condition × category`,
   `scope × category`, `hop × category`.
4. Produce one worked example per category (the prediction closest to the
   median F1 within that bucket).
5. Ship a **Cohen's κ reliability report** from a 30-case human
   spot-check so the classifier's labels can be cited with confidence.

## 3. Categories

### Branch A — Model-level failures

| Code | Label | Trigger |
|------|---|---|
| **A1** | Retrieval miss | Gold answer content is *not* in the top-k retrieved chunks. |
| **A2** | Generation hallucination | Top-k contains the gold, but the prediction asserts entities/numbers not present. |
| **A3** | Tersification artefact | Prediction is a substring of gold (or gold is a substring of prediction) and F1 < 0.5. Measurement-only failure. |
| **A4** | IDK-when-answerable | Prediction matches the IDK regex even though gold content is in the top-k. |
| **A5** | Parse error | `parse_error` on the row is non-null (model produced non-JSON output). |

### Branch B — Corpus-level limits

| Code | Label | Trigger |
|------|---|---|
| **B1** | Fact absent from 10-K | No 10-K filing in the corpus contains the gold, regardless of retrieval. |
| **B2** | Forward-looking unknown | Question scope is `forward_looking`; gold depends on forecast/hint material in 10-K Item 1A or 7. |
| **B3** | Fiscal/calendar mismatch | Question scope is `fiscal_vs_calendar`; answer depends on fiscal-year boundary disambiguation. |
| **B4** | Out-of-scope ticker/year | Question's tickers or years fall outside our 10×6 corpus cell. |
| **B5** | Cross-filing required | `hop_count ≥ 3` AND `scope ∈ {cross_company, inter_year}` — answer requires chaining across filings that L0/L1 can't co-retrieve. |

### Non-failure label

| Code | Label | Trigger |
|------|---|---|
| **NF** | Non-failure | Either (a) F1 ≥ 0.5, or (b) model produced IDK *and* gold content is genuinely not in the top-k (correct abstention). |

Every prediction carries exactly one **primary_cause** and an optional
**secondary_cause** (e.g. a double-fail that is both A2 and B1).

## 4. Architecture

Three sequential stages, each a standalone script. Later stages depend
on output of earlier stages (via files on disk, not function calls —
reproducibility).

```
  predictions.jsonl  (7 models × 4 conditions × ~214 QIDs after vet)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1:  scripts/classify_failures_rules.py                │
│                                                             │
│ Inputs : all predictions, 10-K chunks, filtered KG triples  │
│ Labels deterministically: A3, A4, A5, B2, B3, B4, B5, NF    │
│ Leaves "ambiguous" rows without primary_cause.              │
│ Output : data/eval/failure_taxonomy/rules_stage.jsonl       │
│ Cost   : free, ~30 seconds single-threaded                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2:  scripts/classify_failures_llm.py                  │
│                                                             │
│ Inputs : rules_stage.jsonl + chunk lookup                   │
│ For each row where primary_cause is null:                   │
│   - Render prompt: (Q, gold, prediction, top-k chunks)      │
│   - Call gpt-4o-mini (temp=0, deterministic)                │
│   - Expect JSON: {primary, secondary, reason}               │
│   - Retry with backoff on 429; skip row on persistent 5xx   │
│ Output : data/eval/failure_taxonomy/classified_predictions  │
│          .jsonl                                             │
│ Cost   : ~40-60% of rows hit LLM (~2,400 / 6,000 calls)     │
│          at gpt-4o-mini ≈ $0.15-0.30                        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3:  scripts/classify_failures_agg.py                  │
│                                                             │
│ Inputs : classified_predictions.jsonl                       │
│ Computes:                                                   │
│   - by_model.csv, by_condition.csv, by_scope.csv,           │
│     by_hop.csv    (count matrices)                          │
│   - examples.md   (1 representative Q per category)         │
│   - report.md     (headline + narrative + κ reliability     │
│                    placeholder, filled after kappa run)     │
│ Output: data/eval/failure_taxonomy/*.csv, *.md              │
└─────────────────────────────────────────────────────────────┘

Optional Stage 4: kappa_sample.py
  - Stratified 30 predictions (6 from each of the 5 LLM-decided
    categories: A1, A2, B1, and two most-frequent from Stage 1).
  - Writes data/eval/failure_taxonomy/kappa_sample.jsonl
  - User hand-labels (accept / change / leave blank) via simple CLI.
  - Computes Cohen's κ; writes to report.md.
```

### Stage 1 rule details

- **A3 Tersification:** both directions of substring match, normalized
  (lowercase, punctuation-stripped). Additional guards:
  - Prediction must not match IDK regex (avoids double-labeling A4-then-A3).
  - `min(len(pred), len(gold)) / max(len(pred), len(gold)) >= 0.3` —
    the shorter string must be at least 30 % the length of the longer,
    rejecting trivially short "match as substring" cases.
- **A4 IDK when answerable:** prediction matches
  `r"(?i)i\s*don[\'’]?t\s*know"` AND at least one entity from the
  gold appears (case-folded substring) in at least one retrieved chunk's
  text. "Entity" here = any 3-plus-word n-gram from gold, lowercased,
  punctuation-stripped; reject n-grams that are purely stop-words.
- **A5 Parse error:** `row["parse_error"] is not None`.
- **B2 Forward-looking:** `row["scope"] == "forward_looking"` (labeled at
  QA-set build time).
- **B3 Fiscal/calendar:** `row["scope"] == "fiscal_vs_calendar"`.
- **B4 Out-of-scope:** QA record's `tickers` not all in corpus tickers
  OR `years` not all in corpus years.
- **B5 Cross-filing:** `hop_count >= 3` AND `scope in
  {cross_company, inter_year}`.
- **NF:** `F1 >= 0.5` → NF. Also: `IDK regex match` AND `no gold entity
  in any retrieved chunk` → NF (correct abstention).

When multiple rules match, precedence is: A5 > NF > A3 > A4 > B4 > B5 >
B2 > B3. Rule-matched rows skip Stage 2 entirely.

### Stage 2 LLM prompt skeleton

```
You are classifying why a QA system failed on a 10-K question.

INPUT:
  Question: {Q}
  Gold answer: {gold}
  Model prediction: {pred}
  Top-k retrieved chunks:
    [chunk_1] ...
    [chunk_2] ...
  F1 score: {f1}

CATEGORIES:
  A1 retrieval_miss
    Gold fact is NOT in any top-k chunk.
  A2 hallucination
    Gold fact IS in top-k, but prediction asserts something else
    (wrong number, wrong entity, invented fact).
  B1 corpus_limit
    The gold fact is NOT something any 10-K filing would normally
    contain, regardless of retrieval.
    Examples: stock price on a specific date, analyst opinions,
    board-meeting minutes, forward-looking specifics beyond MD&A.

DECIDE: primary ∈ {A1, A2, B1}
OPTIONAL: secondary ∈ {A1, A2, A3, A4, B1, B2, B3, B4, B5} — include
  iff a second distinct failure mode also applies.

Respond with ONLY JSON:
{"primary": "A1|A2|B1", "secondary": "<code>|null", "reason": "<≤25 words>"}
```

### Cohen's κ details

- Stratified sample: 6 random rows per (A1, A2, B1, and the two most
  populated Stage-1 categories) — aims for 30 total.
- User reviews each row (Q, gold, prediction, chunks) and picks
  a category using a simple CLI prompt.
- Compute κ over the N ≤ 30 human-labeled rows using
  `sklearn.metrics.cohen_kappa_score` (if sklearn is available; else
  inline implementation).
- Interpretation band embedded in the report (Landis & Koch 1977):
  0.01-0.20 slight / 0.21-0.40 fair / 0.41-0.60 moderate / 0.61-0.80
  substantial / 0.81-1.00 almost perfect.
- Reliability threshold for reporting: **κ ≥ 0.4** (moderate). If
  κ < 0.4, refine Stage 2 prompt and re-run.

## 5. Inputs & Outputs

### Inputs (already on disk or pending)

| Path | Rows | Role |
|---|---|---|
| `data/eval/vanilla/<model>/predictions.jsonl` | 129 × 7 = 903 | L0 predictions per model |
| `data/eval/timefilter/<model>/predictions.jsonl` | 903 | L1 |
| `data/eval/kg2rag/<model>/predictions.jsonl` | 903 | L2 (just finished) |
| `data/eval/temporag/<model>/predictions.jsonl` | 903 | L3 (pending ~30 min) |
| `data/samples/10k_chunks.jsonl` | 7,467 | chunk text lookup |
| `data/qa/home_grown.jsonl` + `multihop_filtered.jsonl` | 129 | scope/hop labels |
| (optional) `data/qa/synth_multihop_v1.jsonl` | ~85 after vet | adds to QA after sub-(d) vet |

### Outputs

| Path | Content |
|---|---|
| `data/eval/failure_taxonomy/rules_stage.jsonl` | Post-Stage-1 rows with partial labels. |
| `data/eval/failure_taxonomy/classified_predictions.jsonl` | Final per-prediction labels: `{question_id, model, condition, primary_cause, secondary_cause, reason, source}`. |
| `data/eval/failure_taxonomy/by_model.csv` | Count matrix rows=models, columns=categories. |
| `data/eval/failure_taxonomy/by_condition.csv` | rows=conditions (L0/L1/L2/L3). |
| `data/eval/failure_taxonomy/by_scope.csv` | rows=scope (intra/inter_year/cross_company/fiscal/forward_looking). |
| `data/eval/failure_taxonomy/by_hop.csv` | rows=hop (1/2/3/4). |
| `data/eval/failure_taxonomy/examples.md` | One worked example per category with Q, gold, prediction, chunks, rationale. |
| `data/eval/failure_taxonomy/kappa_sample.jsonl` | 30 hand-labeled rows plus LLM labels (audit trail). |
| `data/eval/failure_taxonomy/report.md` | Final narrative: headline counts, κ value, key findings, links to CSVs/examples. |

## 6. Integration into Final Submission

| Deliverable | Section / Slide | Content |
|---|---|---|
| Report | §6 Discussion | 2 paragraphs — Branch A (model failures) and Branch B (corpus limits) narratives. Cite `by_condition.csv` to show how L1/L2/L3 shift the distribution. |
| Report | §6.1 Limitations | Note A3 as **measurement artefact**, not defect; note Item 1A 35% fatal as a B1 driver; cite κ reliability. |
| Report | Appendix D | Full category definitions + all four count tables + 10 worked examples. |
| Slide deck | Slide 7 Discussion | Heatmap: condition × category grouped bars (4 conditions × 11 categories). |
| Streamlit | (optional, +15 min) | For any answered question, display its failure-category label. |

## 7. Dependencies

- **L3 sweep** must finish before full Stage 1 can run on all conditions
  (~30 min ETA after launch).
- **Sub-(d) vet + re-eval** adds ~85 more QIDs × 4 conditions × 7 models
  to the input; the classifier runs again on the expanded pool. This is
  why all three stages read/write disk: a second run is cheap.

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM labels disagree with human on ambiguous cases (κ < 0.4) | High — undermines Discussion narrative | Kappa check flags this early; if low, refine Stage 2 prompt and re-run (~$0.30 re-run cost). |
| Stage 2 hits OpenAI rate limit | Medium — slows down | Existing `answer_question` retry pattern with exponential backoff; run sequentially after synth so budgets don't conflict. |
| Rule precedence misorders compound failures | Low | Precedence list locked in §4 Stage 1 rule details; unit-testable. |
| "A3 tersification" heuristic false-positives | Medium — affects narrative | Keep detection strict (substring match, case-folded, length ratio > 0.3); any F1 ≥ 0.5 skipped to NF. |
| Examples chosen by median F1 aren't representative | Low | Provide a small CLI flag to pick by alternative criterion if needed post-hoc. |

## 9. Verification

- **Unit check:** Stage 1 on a hand-crafted predictions file (10 rows, one
  per category) must return expected labels.
- **Integration check:** Run full pipeline on L0 predictions only
  (7 models × 129 Qs = 903 rows) end-to-end; confirm:
  - Total labels = 903 (no dropped rows).
  - Sum of NF + each category = 903.
  - `report.md` renders without broken links.
- **Reliability check:** κ ≥ 0.4 on the 30-case sample.

## 10. Locked Decisions (confirmed during brainstorm 2026-04-25)

- **"Heavy" scope** — reusable script + comprehensive coverage, not a one-off manual analysis.
- **10 categories + NF**, exactly as enumerated in §3.
- **Primary + optional secondary** labels per prediction.
- **Full input scope** — every (model × condition × question) prediction.
- **Stage 2 model:** `gpt-4o-mini-2024-07-18`, temperature 0, max_tokens 200.
- **Reliability:** Cohen's κ on 30 stratified samples, threshold 0.4.
- **Output path root:** `data/eval/failure_taxonomy/`.
- **Script naming:** `classify_failures_rules.py`, `..._llm.py`, `..._agg.py`, `kappa_sample.py` — four scripts, not one.

## 11. Non-Goals

- Not a full prompt-engineering study of the answer model.
- Not a retrieval-quality evaluation in its own right (that's R in
  §5 Results).
- Not generating new failure categories automatically — categories are
  fixed by this spec.
- Not retrofitting labels to pre-compacted session history.
