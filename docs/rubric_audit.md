# Rubric Coverage Audit — TempoRAG-KG Final Submission

**Generated:** 2026-04-25
**Total possible:** 30 points

For each rubric category, this document lists (a) what scores full
points, (b) where the corresponding evidence lives in the submission,
and (c) any residual gap that could cost points.

---

## 1. Introduction (max 5 pts)

> Full points: "Clearly describe the background related work, the
> problem, the solution, the expected results, and the contribution
> in a 5-paragraph structure; well-written; interesting RQ / hypothesis
> / IV / DV"

| Criterion                       | Where                                       | Status |
|---------------------------------|---------------------------------------------|--------|
| 5-paragraph structure           | `proposal/final_report.tex` §1 (¶ Background, ¶ Problem and pivot, ¶ Solution, ¶ Expected results, ¶ Contributions) | ✅ |
| Background + related work       | §1 ¶1 + §2 (Related Work)                   | ✅ |
| Problem statement (well-defined)| §1 ¶2 — discovery-framed pivot to 10-K      | ✅ |
| Solution overview               | §1 ¶3 — explicit $2\times2$ ablation        | ✅ |
| Expected results                | §1 ¶4 — preview of the three findings        | ✅ |
| Contributions enumerated        | §1 ¶5 — 4-item enumeration                  | ✅ |
| Interesting RQ                  | §3 RQ1–RQ4 + RQ4★ headline                  | ✅ |
| **IV / DV explicit**            | §1 ¶3 — IV (4 conditions × 7 models × hop × scope), DV (token-F1 + F1\@answered) | ✅ |
| Hypothesis stated               | §3 H1–H4                                    | ✅ |

**Risk / gap:** none. **Expected: 5/5.**

---

## 2. Related Work (max 4 pts)

> Full points: "Clearly describe the gap of related work; summarize
> well; well-written"

| Criterion                | Where                                                             | Status |
|--------------------------|-------------------------------------------------------------------|--------|
| KG²RAG (Zhu 2025)         | §2 ¶1 — explicit base for L2 condition                          | ✅ |
| TempAgent / GoG          | §2 ¶2 — contrasted vs. our planner-free design                   | ✅ |
| FinReflectKG-MultiHop    | §2 ¶3 — anchor for QA set                                        | ✅ |
| GraphRAG (Edge 2024)     | cited in §1 ¶1                                                    | ✅ |
| Gap statement            | §1 ¶1 — neither KG-RAG nor temporal RAG alone targets multi-hop temporal queries | ✅ |

**Risk / gap:** none. **Expected: 4/4.**

---

## 3. Methodology (max 3 pts)

> Full points: "Include all subsections, i.e., Datasets, Preprocessing,
> Models, Training, Experimental Design, Evaluation, with correct
> information inside. The choice and rationale of design is clearly
> rationalized."

| Required subsection | Where in §4               | Status |
|---------------------|---------------------------|--------|
| Datasets            | §4.1 (Corpus + QA bank)   | ✅ |
| Preprocessing       | §4.2 (chunking + embedding + KG extraction + filter) | ✅ |
| Models              | §4.3 (Table 4: 7 models × 3 providers, with role) | ✅ |
| Training            | §4.4 ("no fine-tuning" — explicit non-training paragraph) | ✅ |
| Experimental Design | §4.5 (4 conditions, Figure 1 pipeline, Figure 2 link-jumping) | ✅ |
| Evaluation          | §4.6 (Token-F1, Coverage, F1@answered, bootstrap CI) | ✅ |
| Cost rationale      | Table 3 (cost breakdown non-KG vs. KG) | ✅ |
| Design rationale    | §4.5 (why $k=5$, why $k_{\text{seed}}=3$, why hard mask), §4.4 (why no training) | ✅ |

**Risk / gap:** none. **Expected: 3/3.**

---

## 4. Result (max 3 pts)

> Full points: "Contains figures/graphs; explain the results clearly
> with clear subsections."

| Criterion                       | Where                                              | Status |
|---------------------------------|----------------------------------------------------|--------|
| Subsections                     | §5.1 Overall, §5.2 By hop, §5.3 By scope          | ✅ |
| Figures / graphs                | Figure 1 (pipeline), Figure 2 (link-jumping), 6 PNG figures in `docs/figures/` | ✅ |
| Headline tables                 | Table 5 (4-condition × 7 model overall), Table 6 (by hop), Table 7 (by scope) | ✅ |
| 95\% bootstrap CIs              | Appendix Table 11 + every cell in §5 prose         | ✅ |
| Results explained               | each subsection has prose framing the table        | ✅ |

**Risk / gap:** the per-model F1 table strips CIs from the headline
to keep it on one row; the supplementary Appendix Table 11 has the full
CIs. If grader looks only at the headline, this could read as
"missing CIs" — but the appendix is one paragraph away.
**Expected: 3/3.** Worst case: 2/3 if grader doesn't look at the appendix.

---

## 5. Discussion (max 3 pts)

> Full points: "Discuss important points regarding results and
> hypotheses; insights, and limitations."

| Criterion                       | Where                                              | Status |
|---------------------------------|----------------------------------------------------|--------|
| Results discussion              | §6 ¶1 + §6.1 (failure taxonomy headline + per-condition counts) | ✅ |
| Hypothesis revisit              | §3 H1/H2/H3/H4 are individually marked confirmed/refuted in revised §3 | ✅ |
| Insights                        | §6 — A4 IDK as generation ceiling; L2 regression mechanism via taxonomy | ✅ |
| Limitations                     | §6.2 (5 numbered threats to validity)              | ✅ |
| Reliability                     | §6 Reliability + Appendix tax — explicit LLM-vs-LLM framing | ✅ |
| Per-axis count tables           | Table 8 (headline), Table 9 (by condition)          | ✅ |
| Worked example per category     | Appendix D references `examples.md` produced by aggregator | ✅ |

**Risk / gap:** none. **Expected: 3/3.**

---

## 6. Presentation (max 3 pts)

> Full points: "Well prepared and …" (description truncated in original
> rubric but typically interpreted as: well-organized, fluent, audience-aware)

| Criterion                       | Where                                              | Status |
|---------------------------------|----------------------------------------------------|--------|
| Slides                          | `docs/final_slides.pptx` (T15, pending)             | 🟡 pending |
| Video clip ≤10 min              | `docs/video.mp4` (T17, pending)                     | 🟡 pending |
| Coherent narrative across       | Slides → Video → Report all hit the three findings + A4 | 🟡 pending |
| Voiceover quality               | (user-recorded, T17)                               | 🟡 pending |

**Risk / gap:** depends on T15 + T17. **Expected: 2-3/3** depending on
video polish.

---

## 7. Quality of Code (max 3 pts)

> Full points: (description not visible in rubric image, typically:
> well-organized, tested, documented, reproducible)

| Criterion                       | Where                                              | Status |
|---------------------------------|----------------------------------------------------|--------|
| Repository                      | git repo with feature branches merged to `main`     | ✅ |
| Tests                           | `tests/` — 130 unit + integration tests passing     | ✅ |
| Documentation                   | `README.md` (T18, pending rewrite)                 | 🟡 pending |
| Reproducibility                 | every script has fixed seed + cache; commands documented | ✅ |
| Type / docstrings               | every script has module docstring + function annotations | ✅ |
| Modularity                      | `src/` (taxonomy, retrieval, eval, cache) vs.\ `scripts/` (drivers) | ✅ |
| CI                              | manual (no GH Actions) — small project, acceptable | 🟡 -ish |

**Risk / gap:** README rewrite (T18) is the single largest dependency.
**Expected: 2-3/3.** With T18 done: 3/3.

---

## 8. Creativity (max 3 pts)

> Full points: (typical interpretation: novel approach, original analysis,
> non-trivial contribution)

| Creative angle                  | Where                                              | Notes |
|---------------------------------|----------------------------------------------------|-------|
| L2 negative finding             | §5 + §6                                            | ✅ unusual: most papers do not publish that their proposed addition regresses |
| Failure taxonomy classifier     | §6 + Appendix D + `scripts/classify_failures_*.py` | ✅ reusable, multi-stage, emits structured artefacts |
| A4 IDK as generation-ceiling discovery | §6 ¶2                                       | ✅ reframes "RAG is a retrieval problem" claim |
| Streamlit demo with live KG-flag inspection | §4 + UI                                | ✅ interactive demonstration, not a static screenshot |
| Cost transparency               | Table 3                                            | ✅ explicit non-KG vs. KG breakdown |
| 4-axis auto-vet of synthetic QA | `scripts/auto_vet_synth.py`                        | ✅ multi-criterion LLM judge |

**Risk / gap:** none. **Expected: 3/3.**

---

## 9. Demonstration (max 3 pts)

> Full points: video clip + working demo

| Criterion                       | Where                                              | Status |
|---------------------------------|----------------------------------------------------|--------|
| Streamlit app working           | `app/streamlit_app.py` — running at :8501          | ✅ |
| 4 conditions accessible         | radio button + year-multiselect                     | ✅ |
| Retrieved chunks visible        | per-chunk expander with score + KG-expanded flag    | ✅ |
| F1 vs gold (when matched)        | metric panel                                       | ✅ |
| KG stats sidebar                | KG-look-like answer for TA feedback #1(i)           | ✅ |
| Video clip                      | T17 — pending user voiceover                       | 🟡 pending |
| Recorded screen capture quality  | `frontend-ui-engineering` polish before T17         | 🟡 pending |

**Risk / gap:** video polish + UI tweaks before recording.
**Expected: 3/3** if recording is clean; 2/3 if not.

---

## Roll-up

| Category          | Max | Best case | Worst case (current trajectory) |
|-------------------|-----|-----------|---------------------------------|
| Introduction      | 5   | 5         | 5                               |
| Related Work      | 4   | 4         | 4                               |
| Methodology       | 3   | 3         | 3                               |
| Result            | 3   | 3         | 2 (if appendix-blind grading)   |
| Discussion        | 3   | 3         | 3                               |
| Presentation      | 3   | 3         | 2 (depends on T15+T17)          |
| Quality of Code   | 3   | 3         | 2 (without README rewrite)      |
| Creativity        | 3   | 3         | 3                               |
| Demonstration     | 3   | 3         | 2 (without UI polish + video)   |
| **Total**         | 30  | **30**    | **26**                          |

**Critical-path tasks to keep score at 30:**
1. T15 slides — pin Presentation 3/3
2. T17 video — pin Demonstration 3/3
3. T18 README — pin Code 3/3
4. `frontend-ui-engineering` polish on Streamlit — pin Demonstration 3/3
5. (Already done) every section above marked ✅

**Recommended order to lock score from 26 → 30:**
- (1) `agent-skills:review` on the codebase → fixes Code 3/3 + ammunition for Creativity
- (2) `frontend-ui-engineering` on Streamlit → fixes Demonstration 3/3
- (3) T15 slides → fixes Presentation 3/3
- (4) T16 video script → enables T17
- (5) T17 video record → fixes Demonstration 3/3 (final lock)
- (6) T18 README rewrite → fixes Code 3/3 (final lock)
