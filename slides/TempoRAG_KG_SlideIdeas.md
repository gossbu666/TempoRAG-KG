# TempoRAG-KG — Slide Ideas & Presentation Guide
**NLP Course Project Proposal | AIT 2026**
**Duration: 7 minutes | 8 slides**

---

## Slide 1 — Hook (0:45)

**Title:** Who was the CEO of Apple in 2005?

**Visual:** Single large question on screen — nothing else

**What to say:**
> "Simple question. But if you ask a state-of-the-art KG-RAG system, it retrieves two answers — Steve Jobs AND Tim Cook — simultaneously. It has no idea which one is correct for 2005. This is not a bug. It is a fundamental design flaw that affects 35% of all HotpotQA questions. And nobody has fixed it yet."

**Key point:** Let the question sit on screen for 2-3 seconds before speaking. The silence creates the hook.

---

## Slide 2 — The Problem (1:00)

**Title:** KG-RAG systems treat all facts as permanently valid

**Visual:** Figure 1 — Motivating example (two-column comparison)

```
KG²RAG retrieves:                  TempoRAG-KG retrieves:
─────────────────                  ──────────────────────
(Steve Jobs, CEO, Apple)           (Steve Jobs, CEO, Apple,
(Tim Cook,  CEO, Apple)             valid_from=1997, valid_to=2011)

→ Conflicting — hallucinates       Filter: 1997 ≤ 2005 ≤ 2011 ✓
                                   → Correct: Steve Jobs
```

**What to say:**
> "KG²RAG — published at NAACL 2025 — builds a knowledge graph from documents and traverses it to retrieve connected facts. It works well for multi-hop reasoning. But every fact in the graph is treated as permanently true. No timestamp. No validity window. When you ask about 2005, it retrieves both CEOs and guesses."

---

## Slide 3 — The Gap (0:45)

**Title:** Nobody has solved temporal validity for general-domain KG-RAG

**Visual:** EDA results table + MuSiQue hop breakdown bar

```
┌──────────────────────────────────────────┐
│  Dataset      Any Temporal    Key finding │
│  HotpotQA        35.0%        (2,593 Qs) │
│  MuSiQue         38.3%        (926 Qs)   │
│                                           │
│  MuSiQue by hop:                         │
│  2-hop → 30.2%                           │
│  3-hop → 47.2%  ↑ increases              │
│  4-hop → 46.7%    with complexity        │
└──────────────────────────────────────────┘
```

**What to say:**
> "Papers like TempAgent and EvoReasoner work on temporal KG reasoning — but they all require a pre-built structured KG like Wikidata. Nobody has solved this for general-domain unstructured text. Our EDA shows 35% of HotpotQA and up to 47% of complex 4-hop MuSiQue questions require temporal reasoning. The more complex the chain, the more temporal accuracy matters."

---

## Slide 4 — Our Solution (1:30)

**Title:** TempoRAG-KG: Two additions to KG²RAG

**Visual:** Figure 2 — System architecture diagram (highlight blue components)

```
OFFLINE:
Documents → Chunking → [LLM Extraction + valid_from/valid_to] → Neo4j + FAISS
                                ↑ our addition (blue)

ONLINE:
Query → [Temporal Detection] → Seed Retrieval → [Temporal Filter]
            ↑ blue                                    ↑ blue
→ GEAR Beam Search → [GoG Fill-in] → MST → LLM → Answer
                          ↑ blue
```

**What to say:**
> "TempoRAG-KG adds two things on top of KG²RAG. First: when we extract triples, we also ask the LLM to infer validity intervals. Steve Jobs: CEO 1997–2011. Tim Cook: 2011 to present. Second: before graph expansion, we filter. Only facts valid at the query year enter the traversal. The stale fact never gets retrieved. We also add GoG fill-in for when the KG is incomplete — if the graph hits a dead end, the LLM fills in from its own knowledge."

---

## Slide 5 — Experiment Design (1:00)

**Title:** Three research questions, one ablation design

**Visual:** Ablation conditions table with RQ annotations

```
┌─────────────────────┬──────────┬─────┬───────┬──────────────────┐
│ Condition           │ Temporal │ GoG │ Graph │ Answers          │
├─────────────────────┼──────────┼─────┼───────┼──────────────────┤
│ Vanilla RAG         │    ✗     │  ✗  │   ✗   │                  │
│ KG²RAG (baseline)   │    ✗     │  ✗  │   ✓   │  ← floor         │
│ + Temporal only     │    ✓     │  ✗  │   ✓   │  ← RQ2 component │
│ + GoG only          │    ✗     │  ✓  │   ✓   │  ← RQ2 component │
│ Full model (ours)   │    ✓     │  ✓  │   ✓   │  ← ceiling       │
└─────────────────────┴──────────┴─────┴───────┴──────────────────┘

RQ1: What types of temporal failures exist? → Failure taxonomy (4 types)
RQ2: Does temporal tagging improve F1?       → Main results + ablation
RQ3: How accurately can LLMs extract dates?  → Extraction precision study
```

**What to say:**
> "Each row isolates one component — so every F1 improvement can be attributed to a specific design choice. RQ1 diagnoses the problem with a failure taxonomy. RQ2 measures the fix. RQ3 characterizes how reliable the extraction is. Together they form a complete picture."

---

## Slide 6 — What We Will Measure (0:30)

**Title:** Main results — measuring where the gain happens

**Visual:** Main results table with expected direction annotated

```
┌──────────────────┬───────────┬──────────────────────┬─────────────────────────┐
│ Model            │  Overall  │   Temporal Subset     │   Non-temporal Subset   │
│                  │  F1   EM  │    F1          EM     │     F1           EM     │
├──────────────────┼───────────┼──────────────────────┼─────────────────────────┤
│ Vanilla RAG      │ TBD   TBD │   TBD         TBD    │    TBD          TBD     │
│ GraphRAG         │ TBD   TBD │   TBD         TBD    │    TBD          TBD     │
│ KG²RAG           │ TBD   TBD │   TBD         TBD    │    TBD          TBD     │
├──────────────────┼───────────┼──────────────────────┼─────────────────────────┤
│ TempoRAG-KG      │ TBD   TBD │ ↑ gain expected  TBD │  = stable expected  TBD │
└──────────────────┴───────────┴──────────────────────┴─────────────────────────┘
```

**What to say:**
> "We expect gains to concentrate in the temporal subset — that is the proof that temporal filtering works exactly where it should. Non-temporal F1 should stay stable — confirming the filter does not break general retrieval."

---

## Slide 7 — RQ3: Extraction Study (0:30)

**Title:** How accurate is LLM temporal extraction?

**Visual:** Extraction accuracy table with expected difficulty gradient

```
┌────────────────────────┬───────────┬────────┬──────┬──────────────────────┐
│ Expression Type        │ Precision │ Recall │  F1  │ Expected             │
├────────────────────────┼───────────┼────────┼──────┼──────────────────────┤
│ Explicit year          │    TBD    │  TBD   │ TBD  │ ← High (>80%)        │
│   "became CEO in 2011" │           │        │      │                      │
│ Implicit / relative    │    TBD    │  TBD   │ TBD  │ ← Drop (~60%)        │
│   "during his tenure"  │           │        │      │                      │
│ Conflicting dates      │    TBD    │  TBD   │ TBD  │ ← Lowest (~40%)      │
│   two dates, same fact │           │        │      │                      │
├────────────────────────┼───────────┼────────┼──────┼──────────────────────┤
│ Overall                │    TBD    │  TBD   │ TBD  │                      │
└────────────────────────┴───────────┴────────┴──────┴──────────────────────┘
```

**What to say:**
> "This table will tell us the practical ceiling of our system. If implicit extraction drops significantly, we know exactly where to improve next. This is not just evaluation — it is a roadmap for future work."

---

## Slide 8 — Closing (0:45)

**Title:** The question should never be hard for a knowledge system

**Visual:** Return to the Apple CEO question from Slide 1 — this time with the answer

```
"Who was the CEO of Apple in 2005?"

→ Steve Jobs  (valid_from=1997, valid_to=2011)
   Filter: 1997 ≤ 2005 ≤ 2011 ✓
```

**What to say:**
> "We are building on NAACL 2025, testing on two established benchmarks, and contributing the first systematic taxonomy of temporal failures in general-domain KG-RAG. If our hypotheses hold, the takeaway for the field is clear: temporal metadata is not optional. It should be a standard property of every knowledge graph edge. The question we started with should never be a hard question for a knowledge system. TempoRAG-KG makes it easy."

---

## Timing Summary

| Slide | Content | Time |
|---|---|---|
| 1 | Hook — Apple CEO question | 0:45 |
| 2 | The problem — static KG failure | 1:00 |
| 3 | The gap — EDA numbers | 0:45 |
| 4 | Solution — architecture | 1:30 |
| 5 | Experiment design — ablation table | 1:00 |
| 6 | Main results structure | 0:30 |
| 7 | Extraction study table | 0:30 |
| 8 | Closing — return to hook | 0:45 |
| **Total** | | **6:45** |

Buffer ~15 seconds remaining for transitions and Q&A.

---

## Key Presentation Tips

**Do:**
- Pause after the opening question (Slide 1) — let it land
- Point physically at the blue components in Figure 2 when explaining additions
- When showing TBD tables, say "we expect" not "we will show" — intellectual honesty
- The arrow annotations (↑ gain, = stable) carry the argument — refer to them explicitly

**Don't:**
- Read off the slides
- Say "Today I will present..." — start directly with the question
- Apologize for TBD results — frame them as planned experiments with clear hypotheses
- Rush Slide 4 — the architecture is the core contribution, spend the most time here

---

## Grading Rubric Alignment

| Criterion | Where it shows |
|---|---|
| **Creativity (10pts)** | Opening hook + failure taxonomy concept |
| **Critical Thinking (10pts)** | 3 RQs with IV/DV + ablation design + expected direction annotations |
| **Figures (3pts)** | Fig 1 (motivating), Fig 2 (architecture), Fig 3 (taxonomy) + 3 tables |
| **Writing/Presentation (3pts)** | Clean slide structure, one visual per slide |
| **Introduction — Wobbrock (5pts)** | Slides 1–3 follow 5-part structure exactly |
| **Methodology (10pts)** | Slides 5–7 cover datasets, models, experiment, metrics |

