# TempoRAG-KG

**Temporal Knowledge Graph-Augmented Retrieval for Multi-Hop QA over SEC 10-K Filings.**
Course project for AIT NLU, Spring 2026.

[![tests](https://img.shields.io/badge/tests-130%2F130%20passing-brightgreen)](#)
[![python](https://img.shields.io/badge/python-3.9-blue)](#)
[![paper](https://img.shields.io/badge/paper-PDF-red)](proposal/final_report.pdf)
[![demo](https://img.shields.io/badge/demo-Streamlit-orange)](#streamlit-demo)

A 2×2 retrieval ablation crossing a deterministic **temporal year mask** with a
**knowledge-graph walk** over a 25-filing 10-K corpus, evaluated on 7 LLMs and
129 hand-vetted multi-hop questions.

---

## Contents

- [Headline findings](#headline-findings)
- [Streamlit demo](#streamlit-demo)
- [Quick start](#quick-start)
- [What's in the repo](#whats-in-the-repo)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Reproducing the report numbers](#reproducing-the-report-numbers)
- [Research questions and findings](#research-questions-and-findings)
- [Contributions](#contributions)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Team](#team)
- [License](#license)

---

## Headline findings

1. **Temporal year-mask alone (L1) lifts token-F1 universally** — 7-model
   average +15.9 % over vanilla cosine retrieval (L0), with no model-level
   reversal.
2. **KG-augmented retrieval alone (L2) regresses universally** — −6.7 % over
   L0. Graph expansion disperses cosine-strong seeds with entity-adjacent but
   weaker chunks.
3. **Combined (L3 TempoRAG-KG) wins specifically on hop=3** — +0.071 absolute
   F1 over L1 on `gpt-4.1-nano`, the largest single-cell improvement in the
   entire grid. KG structure pays off where the design predicted: multi-hop
   temporal queries.

A reusable failure-taxonomy classifier over 3,607 predictions further reveals
that **41.8 % of all failures are IDK-when-answerable** — the model abstains
while the gold-bearing chunk is in its top-_k_. This locates the next bottleneck
in **generation, not retrieval**.

📄 Full Report: [`proposal/final_report.pdf`](proposal/final_report.pdf) (12 pages)

---

## Streamlit demo

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```
opens at `http://localhost:8501`. Type any question (or click a sample), pick a
condition (L0 / L1 / L2 / L3), and either run a single condition or click
**“Compare all 4 conditions”** to stack every condition's answer side-by-side
with cached predictions, F1 against gold (when the question matches the
labelled bank), and KG-expansion badges on retrieved chunks.

Sample question that shows the L0/L1/L2/L3 divergence cleanly: _"Which company
had higher data-center revenue in FY2024, NVIDIA or Intel?"_ with
year filter `[2024]`. L0 / L2 say "Intel not provided"; L1 / L3 retrieve the
INTC FY2024 chunk and answer "$12,817 million".

---

## Quick start

```bash
# 1. Python environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env
# add your OPENAI_API_KEY and (optionally) OPENROUTER_API_KEY / GROQ_API_KEY

# 3. Run tests (no network calls, ~1 second)
PYTHONPATH=. pytest tests/ -q
# expected: 130 passed
```

The repository ships with cached `predictions.jsonl`, `summary.json`, the
filtered KG (`data/kg/filtered/triples.jsonl`), embeddings, and the corpus
chunks, so most numbers in the report can be reproduced without paying for
LLM calls.

```bash
# Reproduce the headline tables in <30 s, no API calls:
PYTHONPATH=. python scripts/build_eval_summary.py
PYTHONPATH=. python scripts/build_question_difficulty.py
PYTHONPATH=. python scripts/build_a4_analysis.py
```

---

## What's in the repo

| Path | What lives there |
|---|---|
| `app/streamlit_app.py` | Single-page demo with compare-all-conditions mode |
| `src/` | Library modules (no I/O orchestration): `retrieval.py` (L0–L3 retrievers + KG index), `taxonomy.py` (failure-classifier helpers), `eval.py` (Token-F1 + bootstrap CI), `answer.py` (7-model registry + answer prompt rendering), `cache.py`, `kg_extract.py`, `parse_10k.py` |
| `scripts/` | Operational drivers: `download_10k.py`, `build_embeddings.py`, `run_full_extract.py`, `filter_kg_triples.py`, `run_{vanilla,timefilter,kg_eval}.py`, `synth_multihop_qa.py`, `verify_hop_count.py`, `auto_vet_synth.py`, `classify_failures_{rules,llm,agg}.py`, `kappa_{sample,llm_judge}.py`, `build_{final_figures,eval_summary,a4_analysis,question_difficulty,link_jumping_figure}.py`, `vet_qa.py` |
| `tests/` | 130 unit + integration tests (rule classifier, taxonomy helpers, F1 metric, sampling, KG extractor parsers, LLM-judge prompt rendering / parser, etc.) |
| `prompts/` | `extract_v1.txt` (KG extraction), `answer_v1.txt` (RAG answer), `classify_failure_v1.txt` (Stage 2 LLM judge), `synth_multihop_v1.txt` (synthetic question generation) |
| `data/samples/10k_chunks.jsonl` | 7,467 item-level chunks (≈1k tokens each) — the canonical corpus |
| `data/embeddings/chunks/embeddings.npy` | 7,467 × 1,536 dense vectors, `text-embedding-3-small`, L2-normalised |
| `data/kg/filtered/triples.jsonl` | 57,718 typed triples with `valid_from`/`valid_to` intervals |
| `data/qa/{home_grown,multihop_filtered}.jsonl` | 129 hand-vetted QA records (50 home-grown + 79 multi-hop) |
| `data/qa/synth_pool_*.jsonl` | Synthetic QA expansion pipeline outputs (raw / hop-verified / strict-vetted) |
| `data/eval/{vanilla,timefilter,kg2rag,temporag}/<model>/` | 28 cached prediction sets + summaries (7 models × 4 conditions) |
| `data/eval/failure_taxonomy/` | Classifier outputs: rules / classified / by-model / by-condition / by-scope / by-hop / examples / κ-sample |
| `proposal/final_report.{tex,pdf}` | Final report (12 pages, ACL style) |
| `proposal/{acl.sty, acl_natbib.bst}` | ACL style files (vendored) |
| `proposal/references.bib` | Bibliography |
| `docs/figures/*.png` | 7 publication-ready figures |
| `docs/{rubric_audit, eval_summary_2026_04_25, a4_idk_analysis, question_difficulty_analysis, demo_insights, ...}.md` | Analysis artefacts and supporting documentation |
| `docs/superpowers/{specs,plans}/` | Brainstorming + implementation plans for the failure-taxonomy classifier |
| `notebooks/` | Exploratory notebooks (samples, pilot, KG overview, baseline-vs-full, RQ results) |

---

## Pipeline at a glance

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────────────┐
│ 25 × 10-K    │ →  │ 7,467 chunks    │ →  │ embeddings.npy         │
│ filings      │    │ (item-level)    │    │ (1,536-d, normalised)  │
└──────────────┘    └─────────────────┘    └────────────────────────┘
                            │                        │
                            ▼                        ▼
                    ┌─────────────────┐     ┌────────────────────────┐
                    │ KG extraction   │ →   │ 57,718 typed triples   │
                    │ (gpt-4.1-nano)  │     │ + valid_from/valid_to  │
                    └─────────────────┘     └────────────────────────┘
                                                     │
                ┌────────────────────┬───────────────┴──────────────┬─────────────────────────┐
                ▼                    ▼                              ▼                         ▼
        ┌──────────────┐    ┌──────────────────┐         ┌───────────────────┐       ┌──────────────────────┐
        │ L0 Vanilla   │    │ L1 TimeFilter    │         │ L2 KG²RAG         │       │ L3 TempoRAG-KG       │
        │ cosine top-k │    │ cosine + year    │         │ cosine seeds +    │       │ L2 + triple-level    │
        │              │    │ mask on chunk.fy │         │ entity expansion  │       │ temporal validity    │
        └──────────────┘    └──────────────────┘         └───────────────────┘       └──────────────────────┘
                ↓                    ↓                              ↓                         ↓
                                ┌──────────────────────────────┐
                                │ Answer prompt (gpt-4o, …)    │
                                │ 7 LLMs × 3 providers         │
                                └──────────────────────────────┘
                                              ↓
                                ┌──────────────────────────────┐
                                │ Token-F1 + 95% bootstrap CI  │
                                │ Failure taxonomy classifier  │
                                └──────────────────────────────┘
```

---

## Reproducing the report numbers

**Without paying for any LLM calls** (everything below uses cached predictions):

```bash
# Per-model L0/L1/L2/L3 table + by-hop / by-scope (gpt-4.1-nano)
PYTHONPATH=. python scripts/build_eval_summary.py
# → docs/eval_summary_2026_04_25.md  (matches Report §5)

# Failure-taxonomy aggregate (matches Report §6.1)
PYTHONPATH=. python scripts/classify_failures_agg.py
# → data/eval/failure_taxonomy/{by_*.csv, examples.md, report.md}

# A4 IDK deep-dive (matches Report §6.1 + Appendix D Table 12)
PYTHONPATH=. python scripts/build_a4_analysis.py
# → docs/a4_idk_analysis.md

# Per-question difficulty (matches Report §6.2)
PYTHONPATH=. python scripts/build_question_difficulty.py
# → docs/question_difficulty_analysis.md  +  data/eval/question_difficulty.jsonl

# All figures (matches Report Figure 1, 2, and the seven PNG figures)
PYTHONPATH=. python scripts/build_final_figures.py
# → docs/figures/*.png
```

**With LLM calls** (≈$5 total, ~6 hours wall-clock; not required to verify the
report's numbers):

```bash
# 1. Re-extract the KG over 7,467 chunks with gpt-4.1-nano  (~$3.50, 6 hr)
PYTHONPATH=. python scripts/run_full_extract.py --provider openai --model gpt-4.1-nano
PYTHONPATH=. python scripts/filter_kg_triples.py

# 2. Re-run the four retrieval × seven model sweeps  (~$1.20, 1 hr)
PYTHONPATH=. python scripts/run_vanilla_eval.py
PYTHONPATH=. python scripts/run_timefilter_eval.py
PYTHONPATH=. python scripts/run_kg_eval.py --condition kg2rag
PYTHONPATH=. python scripts/run_kg_eval.py --condition temporag_kg

# 3. Re-classify failures  (~$0.30)
PYTHONPATH=. python scripts/classify_failures_rules.py
PYTHONPATH=. python scripts/classify_failures_llm.py
PYTHONPATH=. python scripts/classify_failures_agg.py
```

All scripts are idempotent (cached on `(model, prompt, params)`) — re-runs that
hit the cache are free.

---

## Research questions and findings

| RQ | Hypothesis | Verdict |
|---|---|---|
| **RQ1** Where does KG²RAG fail? | KG²RAG improves over L0 on hop≥2 but leaves a temporal gap | **Partially refuted** — L2 universally regresses (−6.7 %) instead of improving |
| **RQ2** Does TempoRAG-KG lift over KG²RAG? | L3 > L2, largest gain on inter\_year and fiscal\_vs\_calendar | **Confirmed** — L3 beats L2 in every model; hop=3 jumps +0.071 over L1 |
| **RQ3** Is the temporal KG accurate? | High temporal-interval accuracy → graph coverage bounds L3 ceiling | **Confirmed** — 88.2 % `temporal_type=explicit`, 89.2 % non-null `valid_from` |
| **RQ4★** Can a small LM with retrieval approach a large LM without it? | Temporal grounding narrows the gap | **Partially confirmed** — `gpt-4.1-nano` at L3 (0.253) exceeds `gpt-4o` at L0 (0.183); smaller open models close the gap by ≈30 % |

Full prose, IV/DV/H statements, and per-cell numbers in
[`proposal/final_report.pdf`](proposal/final_report.pdf) §3 and §5.

---

## Contributions

1. A 7,467-chunk SEC 10-K corpus and 129-item QA bank explicitly engineered for
   temporal multi-hop evaluation, with a 60,436-triple KG carrying
   `valid_from`/`valid_to` intervals.
2. A complete 2×2 ablation across 7 LLMs and 3 providers showing that the
   temporal mask is universally helpful, KG²RAG alone is universally harmful on
   this corpus, and the combined L3 condition wins specifically on hop=3.
3. A reusable four-stage failure-taxonomy classifier (rule pass → LLM judge →
   aggregator → κ reliability check) that surfaces _IDK-when-answerable_ as the
   dominant failure mode and re-frames the next-step work in generation, not
   retrieval.
4. A one-page Streamlit demo that exposes the full pipeline (any question, any
   model, any condition) with retrieved-chunk inspection and a live KG-expansion
   flag, so the system can be reproduced and interrogated end-to-end.

---

## Limitations

Five threats to validity, fully discussed in
[`proposal/final_report.pdf`](proposal/final_report.pdf) §6.1:

1. **Small per-cell QA counts.** The 129-item QA set is hand-vetted but small;
   `forward_looking` (_n_=2) and `fiscal_vs_calendar` (_n_=5) cells cannot
   support confident sub-claims on their own. A synthetic expansion pipeline
   produced 4 strict-vetted candidates that we chose not to fold into the
   evaluation (would not move CIs by > ±0.005).
2. **Item 1A coverage gap.** The KG extraction prompt expects
   entity-relation triples; SEC 10-K Item 1A (Risk Factors) is narrative
   prose, leading to a 35.4 % fatal rate on Item 1A chunks. We accept this
   gap rather than retune the prompt; alternative narrative-aware prompts
   are left as future work.
3. **Tersification artefact in fiscal\_vs\_calendar.** The L1 regression on
   that scope (−0.079, _n_=5) is dominated by SQuAD-style token-F1
   penalising correct-but-terse answers against verbose gold text, not by
   retrieval failure.
4. **No prompt-level mitigation tested for A4 IDK.** All seven models share
   one answer prompt; we did not test interventions like
   "attempt-then-flag-uncertainty" that would convert IDKs into low-confidence
   answers.
5. **Corpus scope.** Ten technology issuers across six fiscal years; results
   should not be extrapolated to other sectors or to filings outside this
   window without further evaluation.

Reliability sample (κ between two LLM judges) is reported in Report §6.1 and
Appendix D as `LLM-vs-LLM` agreement; we explicitly do **not** claim it as
human inter-rater reliability.

---

## Citation

If you use TempoRAG-KG in your research or coursework, please cite the team's
final report:

```bibtex
@techreport{kompayak2026temporag,
  title  = {{TempoRAG-KG}: Temporal Knowledge Graph-Augmented Retrieval
            for Multi-Hop Question Answering over SEC 10-K Filings},
  author = {Kompayak, Supanut and Jaemyaem, Aphisit and
            Niamsa-ard, Dechathon and Htet, Kaung Hein},
  year   = {2026},
  institution = {Asian Institute of Technology, NLU Course},
  note   = {Course project final report. \url{https://github.com/gossbu666/TempoRAG-KG}}
}
```

---

## Acknowledgments

- **SEC EDGAR** for the public-domain 10-K filings used as the evaluation
  corpus.
- **Kishore et al. (2025)** —
  [`FinReflectKG-MultiHop`](https://arxiv.org/abs/2510.02906) —
  for the multi-hop QA seed set (79 / 129 items derive from a filtered subset).
- **Zhu et al. (2025)** — KG²RAG — for the foundational graph-walk retriever
  design we extend at L2 / L3.
- **Edge et al. (2024)** — GraphRAG — for framing the role of
  community-detection in graph-based retrieval.
- **AIT NLU course staff** for project guidance and TA feedback that drove
  the v1 → v2 pivot to a temporally-grounded corpus.
- The seven answer-model providers (OpenAI, OpenRouter, Groq) for free-tier or
  low-cost API access that kept the project within a US$20 student budget.

---

## Team

- **Supanut Kompayak** (st126055)
- **Aphisit Jaemyaem** (st126130)
- **Dechathon Niamsa-ard** (st126235)
- **Kaung Hein Htet** (st126477)

Asian Institute of Technology — NLU course, Spring 2026.

---

## License

Course project — internal use.
