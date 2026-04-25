# Temporal Methods Literature Scan (K4)

**Purpose:** Justify the TempoRAG-KG v2 positioning that **extraction + usage** of temporal information in a RAG pipeline is the contribution (not storage location). Per TA feedback #5 (2026-04-16).

**Last updated:** 2026-04-19

---

## The thesis this scan supports

> **Claim:** Storing validity intervals on KG edges is not itself novel — systems like CronKGQA and Wikidata already timestamp pre-existing triples. What is contribution-worthy is (a) **how the intervals are extracted** from unstructured source text (10-K prose) by a prompted LLM, and (b) **how the intervals are used at retrieval time** to gate the candidate set before the generator sees it. The layered ablation (temporal-alone / KG-alone / combined) proves each layer contributes independent lift on 10-K multi-hop QA — a claim no single prior work makes.

The five papers below establish that:

1. **Temporal filtering alone** (no KG) already lifts RAG substantially on temporal QA — **TA-RAG** shows +27.5 to +28.1pp accuracy (the **L1 citation**).
2. **KG-guided RAG** (KG²RAG) treats all facts as timelessly valid — its retrieval path is the **L2 backbone** we port, and the explicit gap we fill.
3. **Financial KG extraction** is a live area (FinReflectKG), but current systems either do not address temporal or encode it as a filing-year default rather than extracted validity intervals.
4. **Financial multi-hop QA benchmarks** exist (FinReflectKG-MultiHop, FinanceBench), but none isolate the KG contribution from the temporal contribution — the ablation gap our work closes.
5. The combination *extraction-from-prose + edge-level temporal + retrieval-time interval filter, with L1/L2/L3 ablation* on 10-K is, to our reading, underexplored.

---

## Paper 1 — KG²RAG (NAACL 2025)  *[backbone / L2 arm]*

**Zhu, X. et al. (2025).** *"Knowledge Graph-Guided Retrieval Augmented Generation."* NAACL 2025. Repo: [nju-websoft/KG2RAG](https://github.com/nju-websoft/KG2RAG).

**What it does.** Builds a passage-level KG on Wikipedia (via `llama3:8b` single-pass, 2-shot prompt, delimiter-separated triples `<h##r##t>$`), then at query time uses the KG to expand retrieved chunks via seed entity matching + 1-hop expansion. Evaluates on HotpotQA and MuSiQue multi-hop QA.

**Assumption.** All KG facts are timelessly valid — no `valid_from`/`valid_to` on edges. The paper makes no distinction between questions whose answer depends on time (e.g. "CFO in 2020") and those that don't.

**Gap we fill.** KG²RAG gives us the retrieval mechanism (seed expansion, chunk union) but cannot answer time-sensitive questions correctly when the KG contains multiple co-existing facts across different years. We reuse its retrieval path as our L2 arm and add edge-level temporal filtering on top for L3. Its hallucination guard (`if r not in ctx and t not in ctx: continue`) inspired our stronger substring-of-evidence check in `src/kg_extract.py`.

---

## Paper 2 — TA-RAG *"Reading Between the Timelines"* (arxiv 2507.22917)  *[L1 citation]*

**Reading Between the Timelines: RAG for the Time-Aware Domain.** arXiv:2507.22917 (2025).

**What it does.** Attaches timestamp metadata to every chunk in a vector index (no knowledge graph). At query time, extracts the temporal expression from the question, computes an interval, and retains only chunks whose timestamp interval overlaps. Reports +27.5pp accuracy at k=5 and +28.1pp at k=10 versus Naive RAG on temporal questions.

**Assumption.** Timestamps live on chunks, not on KG edges. Three explicit failure modes of vanilla RAG on temporal queries are named: *misalignment* (chunk about the wrong period), *insufficient coverage* (correct period but empty retrieval), *endpoint bias* (retrieval skew to the most-recent or most-abundant period).

**Gap we fill.** TA-RAG proves the L1 claim outright — **temporal filtering on chunks already works, without any KG**. But because it operates at chunk level, it cannot disambiguate within-chunk facts (a single 10-K chunk can discuss FY2021 and FY2022 in adjacent sentences). Edge-level temporal resolves within-chunk ambiguity. Our L3 arm combines TA-RAG's filter idea with KG²RAG's entity-guided retrieval to cover both between-chunk and within-chunk temporal disambiguation.

---

## Paper 3 — FinReflectKG (arxiv 2508.17906)  *[baseline KG arm / extraction reference]*

**FinReflectKG: A Reflection-Agent Framework for Financial Knowledge Graphs.** arXiv:2508.17906 (2025). Dataset: [domyn/FinReflectKG](https://huggingface.co/datasets/domyn/FinReflectKG) (17.5M triples, 743 S&P 100 companies, 2014-2024).

**What it does.** Extracts a structured KG from 10-K filings using a reflection-agent architecture (Extract → Critic → Correct loops with Qwen2.5-72B extractor and Qwen3-32B judge). Defines a closed schema: 22 entity types (ORG, PERSON, LOCATION, PRODUCT, REGULATORY_REQUIREMENT, etc.) and 30+ relations (`subject_to`, `subsidiary_of`, `invested_in`, etc.). Best reflection-mode reports 64.8% schema compliance, 39.1% precision, 35.5% faithfulness.

**Assumption on temporal.** The paper text states temporal is "not addressed," but the released dataset schema *does* include `start_date` and `end_date` as "Month YYYY" strings, plus an `extraction_type` flag. The dominant value of that flag is `"default"` — meaning most triples inherit the filing year rather than carrying LLM-extracted validity. Only a minority carry explicit extracted dates.

**Gap we fill.** FinReflectKG validates that financial KG extraction from 10-Ks is feasible at scale, and gives us a direct baseline arm on the same domain. But its temporal field is a coarse default-to-filing-year — our extraction targets explicit LLM-inferred intervals with a `temporal_type` flag (`explicit`, `relative`, `forward_looking`, `unknown`) distinguishing inference from assumption. Comparison against FinReflectKG as a baseline KG arm (Phase 2 B2) quantifies whether extracted temporal outperforms default-filing-year temporal on the same 5 tickers.

---

## Paper 4 — FinReflectKG-MultiHop (arxiv 2510.02906)  *[nearest QA benchmark]*

**FinReflectKG-MultiHop: A Multi-Hop QA Benchmark over Financial KGs.** arXiv:2510.02906 (2025). 555 public questions over the FinReflectKG corpus.

**What it does.** Constructs 2-hop and 3-hop questions with scope labels: intra-document (48.7%), inter-year same-company (41.6%), cross-company same-year (9.7%). Evaluates with an LLM-Judge on a 0-10 scale. Reports that KG-linked retrieval beats a page-window baseline by +24% LLM-Judge with 84.5% fewer input tokens.

**Finding we lean on.** **Inter-year is the hardest scope** (6.72 LLM-Judge vs 7.47 for intra-document). The authors explicitly flag "future work: extend with more cross-company and multi-year queries."

**Gap we fill.** The benchmark proves the importance of multi-year reasoning on 10-Ks but does not isolate the KG contribution from any temporal structure — their retrieval uses KG only, no temporal filter ablation. Our L1/L2/L3 matrix (Vanilla → Temporal-Vanilla → KG²RAG → Full) separates KG contribution from temporal contribution, closing that analytic gap. We reuse their 555 Qs filtered to our 5-ticker subset as the external-validity slice of our QA set, and we match their LLM-Judge scale for direct comparability.

---

## Paper 5 — FinanceBench *[domain-adjacent QA benchmark]*

**FinanceBench: A New Benchmark for Financial Question Answering.** [Islam et al., 2023]. 150 finance-expert-annotated questions over SEC 10-K filings, with short-form gold answers.

**What it does.** Hand-authored questions across 10 categories (metrics, drivers, risks, etc.) with gold answers and source-passage references. Measures model answer accuracy against expert labels; does not enforce retrieval structure.

**Assumption on temporal.** Temporal is not a design axis. Questions are phrased against specific filings ("In FY2022 10-K, ...") but most are intra-document; inter-year comparisons are rare.

**Gap we fill.** FinanceBench validates that 10-K QA is a meaningful task and that expert-quality gold is obtainable. We treat it as a fallback QA source if the MultiHop filtered subset proves too small, but prefer MultiHop because it explicitly stratifies by temporal scope. FinanceBench also reinforces our choice of LLM-Judge over F1/EM: the paper shows that 10-K answers are frequently multi-sentence and that token-overlap metrics underestimate correct answers.

---

## Synthesis

The five papers taken together establish a three-sided gap that TempoRAG-KG fills.

On one side, **TA-RAG** shows that timestamp metadata on chunks alone — no KG — already lifts RAG substantially on temporal queries, but cannot disambiguate within-chunk facts. On another side, **KG²RAG** shows that KG-guided retrieval improves multi-hop QA on Wikipedia, but treats all facts as timelessly valid and fails on temporal queries where the KG contains multiple valid facts across years. On a third side, **FinReflectKG** and **FinReflectKG-MultiHop** prove that financial KG extraction from 10-Ks is feasible and that multi-hop QA over such KGs is the right benchmark, but they neither extract validity intervals as first-class data (FinReflectKG uses filing-year defaults) nor isolate the KG contribution from any temporal structure in their evaluation.

TempoRAG-KG's contribution is the **composition**: prompted LLM extraction of `[valid_from, valid_to]` from 10-K prose, interval-overlap filtering at retrieval time on KG edges (not chunks), and a layered ablation (Vanilla / Temporal-Vanilla / KG²RAG / Full) that attributes lift to each layer independently. The RQ4 capability-parity angle — does temporal-KG retrieval let a small model (LLaMA-3.1-8B) close the gap with a larger model (GPT-4o-mini)? — is specific to this work and directly motivated by the TA feedback to reframe cost as capability.

---

## Risks surfaced and their status

| Risk | Status after reading |
|---|---|
| A 2024-2025 temporal-RAG paper already does what we do | **TA-RAG is the closest** — does chunk-level temporal, no KG. L2/L3 gap preserved. |
| LLM-based temporal extraction is already standard | FinReflectKG shows LLM extraction for financial KGs *without* explicit temporal — standard on extraction, novel on explicit validity intervals. |
| TKG-QA papers actually extract from prose | Not observed in the five papers read. TKG-QA (CronKGQA, TempoQR — not in the five above) assumes pre-timestamped Wikidata/ICEWS. Our extraction-from-prose claim holds. |
| FinReflectKG-MultiHop is direct prior work | **Closest competitor.** Differentiation: (a) we isolate KG from temporal via L1/L2/L3 ablation which they don't; (b) we use explicit-extracted temporal, they use filing-year default; (c) RQ4 capability-parity angle is unique to us. |

**None of the risks fire as blockers.** The most important watch-item is FinReflectKG-MultiHop — if a subsequent version of that paper adds the L1/L2/L3 ablation before our progress report, we must pivot to the RQ4 capability angle as the primary framing.

---

## References

1. Zhu, X. et al. "Knowledge Graph-Guided Retrieval Augmented Generation." NAACL 2025. [Repo](https://github.com/nju-websoft/KG2RAG).
2. "Reading Between the Timelines: RAG for the Time-Aware Domain." arXiv:2507.22917 (2025).
3. "FinReflectKG: A Reflection-Agent Framework for Financial Knowledge Graphs." arXiv:2508.17906 (2025). [HF dataset](https://huggingface.co/datasets/domyn/FinReflectKG).
4. "FinReflectKG-MultiHop: A Multi-Hop QA Benchmark over Financial KGs." arXiv:2510.02906 (2025).
5. Islam, P. et al. "FinanceBench: A New Benchmark for Financial Question Answering." (2023).
