# Progress Presentation Feedback — 2026-04-20

Raw feedback captured from presentation. **To be addressed in subsequent deliverables.**

## Team-specific feedback (TempoRAG-KG)

1. **Show the knowledge graph.** "How does its knowledge graph look like? How the link-jumping between chunk and the main [node] works. This is like what we want to see."
2. **Unanswerable-question categories.** "Does it have any categories that the model cannot answer? Even based on the business document, 10-K, 10-Q — does it have any answer it cannot answer?"

## General feedback (many teams got)

3. **Question generation + logic.** How are questions generated; make the logic explicit.
4. **Qualitative analysis of NLP.** Pick a critical question, walk through the model's reasoning, discuss.
5. **Error analysis.** Show the distribution of errors (not just aggregate F1 numbers).
6. **Cost of non-KG vs KG.** Concrete cost measurement, not just performance lift.
7. **KG injection vs retrieval model-size split.** "Injection should use a large model, retrieval should use a small model." — is this something we need to look at?

## Clustered for prioritization

| Type | Items | Cost | Notes |
|---|---|---|---|
| **A — narrative / slide additions** (no new experiments) | #1 KG viz, #3 QA generation logic, #4 qualitative deep-dive, #6 cost comparison | Low | Data mostly already exists |
| **B — new analysis on existing data** | #2 unanswerable taxonomy, #5 error distribution | Medium | Requires error bucketing work |
| **C — methodology change** | #7 injection-vs-retrieval model-size split | High | Changes the experimental design |

## Next steps

To be decided in brainstorming: which items go into the **next progress
deliverable** (mid-semester) vs the **final report/presentation**.
