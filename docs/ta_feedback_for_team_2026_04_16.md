# TA Feedback — English Summary for Team

**Meeting:** 2026-04-16 consultation with TA
**Summarized by:** Supanut (team lead), 2026-04-17
**Team:** Aphisit (st126130), Dechathon (st126235), Kaung (st126477), Supanut (st126055)
**Full record:** `docs/ta_consultation_2026_04_16.md`
**New plan:** `tasks/plan.md` (v2)

This is a reference summary so the team is aligned on the pivot. No action needed from you other than reading and flagging disagreements.

---

## TL;DR

The TA gave us **7 feedback points**. Three are drops (things we were going to prove that aren't worth proving), three are adds (things we need to include that we missed), and one reframes the whole story.

**New story (one sentence):** *Small models augmented with temporal KG retrieval can match or close the gap with larger models on temporally-grounded multi-hop QA — demonstrated on 10-K SEC filings.*

**Dataset pivot:** HotpotQA + MuSiQue → **25 10-K filings** (AAPL, MSFT, GOOGL, AMZN, META × FY2019–2023).

**Primary research question:** RQ4 (capability parity between small and large models with temporal KG augmentation) is now the spine. RQ1/RQ2/RQ3 become supporting evidence.

---

## The 7 feedback points

### Drop (3)

1. **Drop the ±1 year tolerance proof.**
   *TA's reasoning:* AI models are probabilistic — trying to prove ±1 year tolerance with statistical evidence from a Wikipedia corpus study doesn't make conceptual sense for a language-model-based system. It's work without payoff.
   *What we do instead:* `tolerance` is a configurable knob with default 0. No proof needed. Mention as hyperparameter in methodology.

2. **Drop the α ≥ 0.70 inter-annotator agreement target.**
   *TA's reasoning:* "α ≥ 0.70" is a magic number we pulled from thin air. Either ground it in a cited baseline study or don't assert a threshold.
   *What we do instead:* We report α as a descriptive statistic without a pass/fail threshold. No target to hit.

3. **Drop the valid/null tagging statistical proof.**
   *TA's reasoning:* Technically computable but yields nothing substantive — it's a data limitation, not a finding.
   *What we do instead:* Just report the distribution in the limitations section.

### Add (3)

4. **Add concrete test scenarios.**
   *TA's reasoning:* We need real examples showing "here's a temporal failure case, here's how TempoRAG-KG fixes it." Otherwise the claim is abstract.
   *What we do:* **E4 task** — produce 10 failure→fix case studies for the progress report (`docs/test_scenarios.md`).

5. **Add a temporal-methods literature scan (3–5 papers).**
   *TA's reasoning:* Storage location (metadata vs edge attribute) is not novel. The real contribution is HOW we extract temporal info and HOW we use it during retrieval. We need to cite existing temporal-method papers to position our work.
   *What we do:* **K4 task** — `docs/temporal_methods_scan.md`. 3–5 papers. Candidates: TempoQR, TIMERS, TKGC literature, TempoAtlas, temporal question answering surveys.

6. **Add business applications framing.**
   *TA's reasoning:* Related to the dataset pivot. 10-K filings are business-relevant; the story should connect to real use cases (financial analyst workflows, compliance, M&A research) rather than pure academic multi-hop.
   *What we do:* Progress report includes a short "applications" paragraph in the intro.

### Reframe (1)

7. **Cost framing → capability framing. RQ4 is the new primary story.**
   *Old framing:* "We can evaluate cheaply by using free-tier models." (Budget-driven, defensive.)
   *New framing:* "If small model + temporal KG ≈ large model alone, then temporal augmentation confers capability — cost reduction is a consequence." (Story-driven, offensive.)
   *What we do:* RQ4 (generator ablation: LLaMA-3.1-8B vs GPT-4o-mini × {Vanilla, KG²RAG, Full}) becomes the primary table in the progress report. RQ2 (overall F1 lift) demoted to supporting.

---

## What we're keeping from v1

- KG²RAG backbone (NAACL 2025) as baseline
- Temporal validity intervals `[valid_from, valid_to]` on triples
- NetworkX in-memory KG (not Neo4j)
- `$20` total budget cap
- Groq LLaMA-3.1-8B as free primary generator
- All infra code (`src/cache.py`, `src/eval.py`, `src/iaa.py`, `src/sampling.py` base)
- EDA results (35% / 38.3% temporal prevalence) as motivation

## What's new

- **Dataset:** 25 10-K filings (see `docs/10k_scoping.md` for details)
- **Question set:** FinanceBench subset + template-synthesized questions (~100 Q total)
- **Extraction prompt:** full rewrite for financial domain (A2)
- **HTML parser:** SEC EDGAR filings (A3)
- **Test scenarios:** 10 failure→fix cases (E4)
- **Lit scan:** 3–5 temporal-method papers (K4)

## Timeline

| Week | Focus |
|---|---|
| Week 1 (now) | Phase 0 adaptation (A1–A4), K2/K4 docs |
| Week 2 | Pilot (P1/P2) → Phase 2 KG build + baseline + annotations (parallel) |
| Week 3 | Phase 3 pipeline + start Phase 4 evaluation |
| Week 4 | Finish Phase 4 + progress report (R1) |

**Progress report deadline:** ~2026-05-15.

## Budget check

See `docs/10k_scoping.md` §10 for the full breakdown. Committed ~$12.55 of $20, ~$7.45 buffer. Main cost items:

| Item | Cost |
|---|---|
| KG build (~2,750 chunks × Gemini Flash) | ~$1.54 |
| RQ4 sweep (GPT-4o-mini on temporal subset) | ~$3.00 |
| RQ2 / RQ3 supporting | ~$2.00 |
| Optional ceiling run (GPT-4o) | ~$6.00 |

## Team task assignments (to fill in on next sync)

`tasks/todo.md` has an assignment table with `?` placeholders. Please claim tasks you want to own. Recommended pairing:

- **A1 + A3** (data layer) — one owner, ~1.5 days
- **A2 + A4** (content / question design) — one owner, ~1.5 days
- **K4** (lit scan) — good for anyone who likes reading, ~1 day
- **B3** (annotation) — needs 2 different people for the 50 passages

## Questions / disagreements?

If any of the above sounds wrong to you, flag it in Discord/whatever-we-use before Monday. Otherwise I'll take silence as agreement and we execute the plan starting this week.

Full TA meeting minutes (all questions we asked and their responses): `docs/ta_consultation_2026_04_16.md`
