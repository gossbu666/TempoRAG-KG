# Final Video Script — TempoRAG-KG

**Duration target:** 10:00 (10 min cap)
**Budget breakdown:** 9:30 narration over slides + 0:30 live Streamlit demo
**Recording flow:** Record screen first (advance slides at the listed timings),
then voiceover separately to match. All cues are wall-clock from 0:00.

**Pacing:** ~150 words/min English narration. Slow down on numbers; speed up
on transitions. If you finish a slide block early, hold the visual — the next
cue is what locks you in.

---

## Slide 1 — Title  (0:00 → 0:15  ·  15 sec)

**On screen:** Title slide. Stay still — no clicks.

> "Hi, I'm Supanut. With Aphisit, Dechathon, and Kaung, this is our final
> project for AT82.05 — Natural Language Understanding. We're presenting
> **TempoRAG-KG**: a temporal-aware, knowledge-graph augmented RAG for
> multi-hop question answering on SEC 10-K financial filings."

---

## Slide 2 — Hook  (0:15 → 0:50  ·  35 sec)

**On screen:** Click to slide 2. Pause briefly on the question quote box.

> "Let's start with a real question. *Which company had higher data-center
> revenue in fiscal 2024 — NVIDIA or Intel?* When we run this through a
> standard vanilla RAG, the system retrieves NVIDIA-heavy chunks because
> they're the most semantically similar — and answers *not provided in
> the excerpts*. The Intel chunk never makes it into the top-5.
>
> What we want is a system that knows two things: the right *year*, and
> the right *entities* — even when the question seeds with a different
> company. Our claim is that you need both. Neither alone closes the
> gap."

---

## Slide 3 — Research Questions & Variables  (0:50 → 1:20  ·  30 sec)

**On screen:** Click to slide 3.

> "We frame this as four research questions. RQ1: does temporal filtering
> alone help? RQ2: does KG entity expansion alone help? RQ3 — the
> headline — does combining temporal and KG outperform either alone?
> And RQ4: where do the lifts come from — retrieval, or generation?
>
> Independent variables: 4 retrieval conditions, 7 LLMs, hop count, scope.
> Dependent variable: token-F1 with 95% bootstrap confidence intervals."

---

## Slide 4 — Related Work  (1:20 → 1:50  ·  30 sec)

**On screen:** Click to slide 4.

> "We organize prior work on a 2-by-2 grid. Vanilla RAG is time-blind and
> entity-blind. Temporal RAG adds year masks but ignores entities.
> KG-squared-RAG adds entity walks but ignores time. The
> *combined* cell — entity-aware and time-aware — is empty in the
> literature. That's the gap we fill, on the 10-K corpus where
> dates are clean by construction."

---

## Slide 5 — Method  (1:50 → 2:20  ·  30 sec)

**On screen:** Click to slide 5.

> "Same data, same prompts, same models — only retrieval differs. L0 is
> vanilla cosine. L1 adds a year mask. L2 swaps the year mask for a KG
> entity walk. L3 combines both. Each cell answers exactly one
> question about which mechanism is doing work. All retrieval is cached
> on the question and condition, so identical inputs always produce
> identical outputs — the experiment is fully reproducible."

---

## Slide 6 — Data & QA Construction  (2:20 → 2:55  ·  35 sec)

**On screen:** Click to slide 6.

> "Our corpus is 10 issuers across fiscal years 2019 to 2024 — 7,467
> chunks pulled directly from SEC EDGAR. We chunk at 1,500 characters
> with 200-character overlap, and extract a knowledge graph using
> gpt-4.1-nano with an explicit temporal schema — every triple has a
> *valid_from* and *valid_to* date.
>
> The 129 questions come from the FinReflectKG-MultiHop benchmark,
> filtered to our coverage. We tried generating synthetic questions too
> — 128 candidates, only 4 passed our 4-axis auto-vet. We dropped them
> and kept that as a methodological lesson."

---

## Slide 7 — KG Visualization  (2:55 → 3:25  ·  30 sec)

**On screen:** Click to slide 7.

> "Quick look at the KG: 57,718 triples, 293 unique subjects, 60 filings,
> averaging 7.7 triples per chunk. Here's a sample subgraph for Apple's
> fiscal 2022 — revenue, segment breakdown, leadership, headquarters.
> The *valid_from* and *valid_to* columns are extracted explicitly,
> per triple — and that's what L3's hard mask uses to drop expired
> facts."

---

## Slide 8 — Link-jumping Mechanism  (3:25 → 4:00  ·  35 sec)

**On screen:** Click to slide 8. Use the figure on the left as the visual anchor.

> "This is how the KG *jumps* between chunks. Step 1: cosine top-3 seeds
> from the question. Step 2: extract subjects and objects from those
> seeds' triples. Step 3: pull every other chunk that mentions any of
> those entities. Step 4: re-rank with cosine, keep top-5.
>
> For L3 only, we add one extra filter — drop any expanded chunk whose
> triples don't overlap the query's year filter. That's the temporal
> hard mask."

---

## Slide 9 — Headline Results  (4:00 → 4:45  ·  45 sec)

**On screen:** Click to slide 9. Pause on the figure for 5 seconds, then walk through the bullets.

> "Three findings. First — TimeFilter beats Vanilla on every single
> model. Plus 5.6 to plus 24.7 percent, averaging plus 15.9. H1 is
> supported.
>
> Second — KG-squared-RAG *alone* loses on every model. Average minus
> 6.7 percent. H2 is refuted: graph walking without a temporal anchor
> injects noise on this corpus.
>
> Third — TempoRAG-KG averages roughly the same as L1 *overall*, but
> on hop-3 questions with gpt-4.1-nano specifically, it gains plus
> 0.071 over L1. The combined design wins exactly where multi-hop
> entity bridging is supposed to help.
>
> And bonus — F1-at-answered stays flat at 0.36 across all conditions.
> The retrieval changes are moving coverage, not answer quality. The
> bottleneck is generation."

---

## Slide 10 — Ablation Table  (4:45 → 5:20  ·  35 sec)

**On screen:** Click to slide 10. Point at the green column (L1) when reading "wins 7/7".

> "The full table. Every row is one model; every green cell is the
> best condition for that row. L1 — TimeFilter — is the best column
> on 7 out of 7. L2 — KG-only — loses on 7 out of 7. The averaged
> story masks L3's targeted hop-3 win, but it's there in the by-hop
> breakdown next.
>
> Beyond this table, we also did by-hop, by-scope, a failure
> taxonomy, a qualitative deep-dive, and a cost analysis — all
> coming up."

---

## Slide 11 — By-hop / By-scope  (5:20 → 5:50  ·  30 sec)

**On screen:** Click to slide 11. Two figures side by side.

> "Hop-1 questions converge — vanilla cosine already finds the single
> chunk; year masks and KG walks add nothing. Hop-3 is where L3
> earns its keep — plus 0.071 on gpt-4.1-nano, exactly the slice the
> design targets. By scope, the *cross-company* questions are where
> L1's year mask carries the headline lift."

---

## Slide 12 — Qualitative Deep-dive  (5:50 → 6:30  ·  40 sec)

**On screen:** Click to slide 12. Walk through the table row-by-row.

> "Back to the opening question — NVIDIA versus Intel data-center
> revenue in 2024. L0 fails: cosine pulls NVIDIA chunks, Intel
> never appears. L1 succeeds: the year mask forces 2024 filings into
> the candidate pool, and Intel's 12.8 billion comes through.
>
> L2 — KG-only — fails *identically* to L0. The cache hits prove it:
> the prompt is byte-for-byte the same. KG entity expansion seeds with
> NVIDIA, walks NVIDIA-related entities, and never bridges to Intel.
> L3 succeeds, identical to L1 — temporal does the work, KG adds
> nothing on this query.
>
> One question, four conditions, mechanism visible in real time."

---

## Slide 13 — Failure Taxonomy  (6:30 → 7:05  ·  35 sec)

**On screen:** Click to slide 13. Point at the dominant A4 bar in the chart.

> "When L3 doesn't win, why not? We classified all 3,607 predictions
> across 10 failure modes — five model-level, five corpus-level.
>
> The dominant failure is A4 — *I-don't-know when answerable*. 41.8
> percent of all predictions. The model has the chunks but refuses
> to commit. A2, *wrong-year retrieval*, drops from 18 percent at L0
> to 6 percent at L1 — that's exactly TimeFilter doing its job.
>
> We report inter-rater reliability honestly: kappa equals 0.200,
> LLM-versus-LLM, not a human IRR substitute. Slight agreement is
> what we actually have."

---

## Slide 14 — Cost  (7:05 → 7:30  ·  25 sec)

**On screen:** Click to slide 14.

> "Total project cost: 8 dollars 90 cents one-time, plus 5 cents per
> uncached demo query. The KG extraction is 92 percent of the bill.
> Eval is free because everything is cached. In production, this is
> trivial — well under any operational budget."

---

## Slide 15 — Lessons & Future Work  (7:30 → 8:05  ·  35 sec)

**On screen:** Click to slide 15.

> "What we learned. One — negative findings matter. L2 regresses
> universally; we report it because the *mechanism* is the contribution.
> Two — token-F1 has a ceiling: zero out of 129 questions hit F1
> 0.8 due to tersification artifacts. Future scoring should target
> entity-sets and canonical numeric values.
> Three — synth QA is hard. Hop-count is verifiable; relevance and
> answerability are not, without humans.
>
> Future work follows our advisor's suggestion: inject-large,
> retrieve-small. Use a big model at extraction once, a small model
> per query. And tackle the A4 ceiling on the generation side —
> uncertainty-aware decoding."

---

## Slide 16 — Live Demo  (8:05 → 8:35  ·  30 sec)

**On screen:** Click to slide 16, immediately switch window to **live Streamlit at :8501**.

**Action sequence (record this once, then narrate):**

```
0:00  Click sample button "Cross-company hop=2 (NVDA vs Intel)"
0:03  Year [2024] auto-fills
0:05  Click "Compare all 4 conditions"  → click "Ask"
0:10  Scroll: L0 ❌  · L1 ✅  · L2 ❌  · L3 ✅  + 🧬 KG-expanded badge
0:22  Hover briefly on retrieved chunks panel — show Intel only in L1/L3
0:28  Hold on the F1 score row
```

**Narration:**

> "Let me show this live. One click — sample question. Compare all four
> conditions. L0 says *not provided*. L1 finds Intel: 12.8 billion.
> L2 fails again. L3 succeeds, with the KG-expanded badge visible.
> Same model, same chunks pool, four retrieval strategies — exactly
> the result the ablation predicts."

---

## Slide 17 — Conclusion  (8:35 → 9:00  ·  25 sec)

**On screen:** Switch back from Streamlit to slide 17.

> "To wrap. Temporal filtering is universally beneficial. KG entity
> expansion *alone* regresses — and that's a finding worth reporting.
> Combined L3 wins where the design predicts it should — hop-3.
> And the next bottleneck isn't retrieval — it's generation.
>
> Code, KG, predictions, and the Streamlit app are all open-sourced
> at github dot com slash gossbu666 slash TempoRAG-KG."

---

## Slide 18 — Thank You  (9:00 → 9:10  ·  10 sec)

**On screen:** Click to slide 18. Hold.

> "Thank you for your time. Happy to take questions."

---

## Total elapsed:  9:10  ·  Buffer: 50 sec

**Buffer use ideas:**
- Slow down on slide 9 (results) by 10 sec — these are the headline numbers
- Add 10 sec for a confidence-interval mention on slide 10
- Add 20 sec on slide 12 to read each row of the qualitative table
- Reserve 10 sec for end-of-clip silence before fade

If running over: cut slide 7 (KG viz) from 30 sec to 20 sec — it's
nice-to-have, not load-bearing.

---

## Recording checklist

**Before recording screen:**
- [ ] Streamlit running at :8501 — pre-warm cache by clicking each sample once
- [ ] Browser zoom 110% so text is readable in 1080p
- [ ] Close all other apps; hide Dock; menu bar clean
- [ ] Quicktime → New Screen Recording → select Streamlit window only
- [ ] Slide deck open in Keynote, presenter mode off (full-screen)

**During recording:**
- [ ] Advance slides at the timings above; if you miss a cue, keep going
  — voiceover will sync to whatever the screen actually shows
- [ ] On slide 16: switch to Streamlit, do the demo, switch back

**Voiceover (recorded separately):**
- [ ] Read this script with watch open; aim for ±2 sec per slide
- [ ] Re-record any slide where you stumble on a number — numbers must be
  exact (15.9%, 6.7%, 0.071, 41.8%, 12.8 billion)

**Post-production:**
- [ ] Combine screen + voice in iMovie or DaVinci Resolve
- [ ] Add 1 sec fade-in at start, 1 sec fade-out at end
- [ ] Export 1080p, MP4, target file size < 100 MB
