## 1. Proposal recap

TempoRAG-KG extends KG²RAG (NAACL 2025) by attaching explicit
`[valid_from, valid_to]` validity intervals to every KG triple, enabling
deterministic temporal filtering during retrieval on multi-hop QA.
Motivation: our EDA shows **35.0% of HotpotQA** and **38.3% of MuSiQue**
dev questions contain temporal expressions, rising from 30.2% in 2-hop
to 46.7% in 4-hop. Original three research questions: (RQ1) what
temporal failure types occur in KG-RAG, (RQ2) does temporal tagging
improve F1 on temporal subsets, (RQ3) how accurately can LLMs extract
validity intervals from unstructured text.

---

## 2. Your feedback (our understanding — please correct if misread)

1. **IAA metric** — We did not specify an inter-annotator agreement
   metric for the RQ3 gold-standard annotation.
2. **±1 year tolerance** — We asserted ±1 year tolerance in temporal
   extraction evaluation without linguistic or empirical justification.
3. **Failure taxonomy gap** — Our 4-type taxonomy misses
   **relative/ambiguous temporal references** such as "recently", "at
   the time", "this quarter" — cases where year cannot be resolved
   without document context.

---

## 3. How we will address each

### 3.1 IAA metric → Krippendorff's α (interval distance)

Year annotations are **ordinal-interval**, not categorical — the
distance between 2010 and 2015 is meaningfully larger than between
2010 and 2011. Cohen's κ would treat all disagreements equally and
mis-weight the data. We adopt Krippendorff's α with interval distance
on the 100-passage gold set, annotated independently by two team
members; disagreements adjudicated. **Target: α ≥ 0.70**.

### 3.2 ±1 year tolerance → empirical justification from Wikipedia

We will ground the tolerance in corpus statistics. Plan: sample ~200
Wikipedia references to dated events and count how often the narrative
year differs from the actual event year by 1 (e.g., *"late 2010"* for
a January 2011 event). Report the off-by-one frequency as the
empirical basis. Decision rule: if off-by-one rate < 5%, we drop the
tolerance; if ≥ 5%, we keep τ=1. Filter equation becomes:

> `(v_f ≤ y_q + τ) ∧ (v_t ≥ y_q − τ)` with null-validity retained.

### 3.3 Failure taxonomy → add Type 3b (Relative/Ambiguous)

Added as a separate row in the taxonomy (now 5 types):

| # | Type | Example |
|---|---|---|
| 1 | Stale Fact | "Jobs is CEO" after 2011 |
| 2 | Conflicting Facts | Both Jobs and Cook returned as CEO |
| 3 | Missing Temporal Context | Cook is CEO, no valid_from known |
| **3b** | **Relative/Ambiguous Temporal Reference** | **"recently", "at the time", "this quarter"** |
| 4 | Temporal Hop Failure | Hop 2 retrieves stale fact |

Extraction rule for Type 3b: set `valid_from = valid_to = null`, tag
`metadata.temporal_type = "relative"`. Triples retained by the
conservative filter but reported separately as a known limitation.

---

## 4. Other revisions (not from your feedback — for your awareness)

**Scope reductions** — hard $20 student-funded budget, 4-person team:

| Dimension | Original | Revised |
|---|---|---|
| HotpotQA eval | 7,405 Q | 1,000 sampled (500T / 500NT) |
| MuSiQue eval | 2,417 Q | 500 stratified by hop |
| RQ3 gold | 200 passages | 100 passages |
| Ablation | 5 conditions | 3 (Vanilla / KG²RAG / Full) |
| Seeds | 3 | 1 + bootstrap CIs |
| KG storage | Neo4j | NetworkX in-memory |
| Context organization | MST | Concatenation v1 (MST deferred) |

**New contribution — RQ4**: Does the benefit of temporal filtering
scale **inversely** with generator capability? Hypothesis: smaller
models benefit more because they have less parametric temporal
knowledge to fall back on. The budget-forced multi-generator setup
(LLaMA-8B / LLaMA-70B / Gemini Flash / GPT-4o-mini — 3 of 4 free on
Groq) *is* the experiment. Constraint → contribution.

---

## 5. Current status and questions for you today

**Completed**: EDA (35% / 38.3%), proposal submitted, revision plan,
GitHub repo scaffolded, cache layer, evaluation harness.
**In progress**: deterministic sampling, IAA script, extraction
prompt.
**Next gate**: 20-chunk pilot on Gemini Flash before full KG build.

**Questions we need your input on:**

1. Do our responses to feedback items 1–3 (§3.1–§3.3) address your
   concerns adequately?
2. Is the scope reduction in §4 acceptable, or do you need us to
   defend statistical power more explicitly?
3. **RQ4** — valuable contribution, or scope creep we should drop?
4. **EvoReasoner** — acceptable to keep as related work only (not
   reproduced baseline) given time and budget?
5. **Primary benchmark** — HotpotQA isn't a purpose-built temporal
   benchmark. Should we stick with it, or switch to TimeQuestions /
   MultiTQ?
