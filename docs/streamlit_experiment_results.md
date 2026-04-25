# Streamlit demo experiments — programmatic run

Run via `scripts/run_demo_experiments.py` on the same retrieval + answer pipeline as `app/streamlit_app.py`. Answer model: `gpt-4o-mini` at temperature 0. Cache shared with the eval pipeline, so re-runs are free.

Each experiment lists:
- the question (verbatim)
- year filter for temporal conditions
- per-condition retrieved chunks (top-5 with cosine scores; KG-expanded chunks tagged for L2/L3)
- the model's answer
- gold + Token-F1 if the question matches a labeled QA record

---

## E1 — Smoke — easy single-hop (all conditions converge)

**Question:** What was Apple's revenue in fiscal 2022?

**Hypothesis:** All four conditions return the same chunks and answer because the gold-bearing chunk dominates cosine.

### L0 (years=—)

Retrieved chunks:

  1. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.722
  2. `AAPL_FY2022_item7_001` (AAPL FY2022 item7) score=0.705
  3. `AAPL_FY2021_item7_001` (AAPL FY2021 item7) score=0.694
  4. `AAPL_FY2024_item8_010` (AAPL FY2024 item8) score=0.687
  5. `AAPL_FY2021_item8_013` (AAPL FY2021 item8) score=0.682

**Answer:**

> $394,328 million

---

### L1 (years=[2022])

Retrieved chunks:

  1. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.722
  2. `AAPL_FY2022_item7_001` (AAPL FY2022 item7) score=0.705
  3. `AAPL_FY2022_item8_007` (AAPL FY2022 item8) score=0.671
  4. `AAPL_FY2022_item8_015` (AAPL FY2022 item8) score=0.661
  5. `AAPL_FY2022_item7_002` (AAPL FY2022 item7) score=0.654

**Answer:**

> $394,328 million

---

### L2 (years=—)

Retrieved chunks:

  1. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.722
  2. `AAPL_FY2022_item7_001` (AAPL FY2022 item7) score=0.705
  3. `AAPL_FY2021_item7_001` (AAPL FY2021 item7) score=0.694
  4. `AAPL_FY2024_item8_010` (AAPL FY2024 item8) score=0.687 · **KG-expanded**
  5. `AAPL_FY2021_item8_013` (AAPL FY2021 item8) score=0.682 · **KG-expanded**

**Answer:**

> $394,328 million

---

### L3 (years=[2022])

Retrieved chunks:

  1. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.722
  2. `AAPL_FY2022_item7_001` (AAPL FY2022 item7) score=0.705
  3. `AAPL_FY2021_item7_001` (AAPL FY2021 item7) score=0.694
  4. `AAPL_FY2024_item8_010` (AAPL FY2024 item8) score=0.687 · **KG-expanded**
  5. `AAPL_FY2024_item8_009` (AAPL FY2024 item8) score=0.667 · **KG-expanded**

**Answer:**

> $394,328 million

---


## E2 — L1 year-mask actually filters (no year in question)

**Question:** What was Apple's revenue?

**Hypothesis:** L0 returns recent chunks (FY2023/FY2024 bias). L1 with year=[2020] forces FY2020 chunk → answer is FY2020 revenue (~$274.5B), not the latest.

### L0 (years=—)

Retrieved chunks:

  1. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.653
  2. `AAPL_FY2021_item8_013` (AAPL FY2021 item8) score=0.646
  3. `AAPL_FY2020_item8_014` (AAPL FY2020 item8) score=0.646
  4. `AAPL_FY2024_item8_009` (AAPL FY2024 item8) score=0.638
  5. `AAPL_FY2023_item8_009` (AAPL FY2023 item8) score=0.636

**Answer:**

> $394,328 million for fiscal 2022

---

### L1 (years=[2020])

Retrieved chunks:

  1. `AAPL_FY2020_item8_014` (AAPL FY2020 item8) score=0.646
  2. `AAPL_FY2020_item7_002` (AAPL FY2020 item7) score=0.626
  3. `AAPL_FY2020_item8_013` (AAPL FY2020 item8) score=0.617
  4. `AAPL_FY2020_item8_048` (AAPL FY2020 item8) score=0.613
  5. `AAPL_FY2020_item7_001` (AAPL FY2020 item7) score=0.596

**Answer:**

> $274,515 million

---


## E3 — L2 KG²RAG graph walk vs L0 cosine

**Question:** Compare Apple's Services revenue to Microsoft's cloud revenue for fiscal year 2022

**Hypothesis:** L0 may favor a single ticker; L2 entity-walk should include chunks from the second ticker via shared entities (KG-expanded flag visible).

### L0 (years=—)

Retrieved chunks:

  1. `MSFT_FY2023_item8_086` (MSFT FY2023 item8) score=0.667
  2. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.659
  3. `AAPL_FY2024_item8_009` (AAPL FY2024 item8) score=0.654
  4. `AAPL_FY2021_item8_013` (AAPL FY2021 item8) score=0.646
  5. `MSFT_FY2019_item8_114` (MSFT FY2019 item8) score=0.645

**Answer:**

> Apple's Services revenue for fiscal year 2022 was $78,129 million, while Microsoft's cloud revenue for fiscal year 2022 was $111.6 billion.

---

### L2 (years=—)

Retrieved chunks:

  1. `MSFT_FY2023_item8_086` (MSFT FY2023 item8) score=0.667
  2. `AAPL_FY2022_item8_014` (AAPL FY2022 item8) score=0.659
  3. `AAPL_FY2024_item8_009` (AAPL FY2024 item8) score=0.654
  4. `AAPL_FY2021_item8_013` (AAPL FY2021 item8) score=0.646 · **KG-expanded**
  5. `MSFT_FY2019_item8_114` (MSFT FY2019 item8) score=0.645 · **KG-expanded**

**Answer:**

> Apple's Services revenue for fiscal year 2022 was $78,129 million, while Microsoft's cloud revenue for fiscal year 2022 was $111.6 billion.

---


## E4 — L3 inter-year (correctly scoped year filter)

**Question:** How did Amazon's AWS operating income evolve from 2020 to 2023?

**Hypothesis:** With a 4-year filter, L3 keeps triples whose validity intersects 2020-2023 → answer covers all 4 years.

### L3 (years=[2020, 2021, 2022, 2023])

Retrieved chunks:

  1. `AMZN_FY2022_item7_012` (AMZN FY2022 item7) score=0.741
  2. `AMZN_FY2023_item7_012` (AMZN FY2023 item7) score=0.725
  3. `AMZN_FY2022_item7_013` (AMZN FY2022 item7) score=0.712
  4. `AMZN_FY2020_item7_013` (AMZN FY2020 item7) score=0.710 · **KG-expanded**
  5. `AMZN_FY2022_item7_011` (AMZN FY2022 item7) score=0.708 · **KG-expanded**

**Answer:**

> AWS operating income was $13,531 million in 2020, $22,841 million in 2022, and $24,631 million in 2023.

---


## E5 — A4 IDK-when-answerable demonstration

**Question:** What risk factors did Meta disclose regarding EU regulations in 2023?

**Hypothesis:** Even with the right Meta FY2023 Item 1A chunks retrieved, gpt-4o-mini abstains with 'I don't know'. Demonstrates the 41.8% A4 finding.

### L1 (years=[2023])

Retrieved chunks:

  1. `META_FY2023_item1A_053` (META FY2023 item1A) score=0.617
  2. `META_FY2023_item1_007` (META FY2023 item1) score=0.608
  3. `META_FY2023_item8_049` (META FY2023 item8) score=0.553
  4. `META_FY2023_item8_048` (META FY2023 item8) score=0.548
  5. `META_FY2023_item1A_054` (META FY2023 item1A) score=0.520

**Answer:**

> Meta disclosed risks related to evolving EU regulations that could impose significant fines, affect user growth and engagement, and complicate data transfer processes, particularly following the invalidation of the Privacy Shield and the IDPC's May 12, 2023 decision that resulted in a EUR €1.2 billion fine for non-compliance with the GDPR.

---


## E6 — L2 regression — KG²RAG ≈ L0 on easy intra query

**Question:** What was Microsoft's revenue in fiscal 2022?

**Hypothesis:** Same chunks, same answer — L2 graph walk doesn't add value when cosine seeds already span the answer.

### L0 (years=[2022])

Retrieved chunks:

  1. `MSFT_FY2023_item8_086` (MSFT FY2023 item8) score=0.689
  2. `MSFT_FY2022_item7_000` (MSFT FY2022 item7) score=0.670
  3. `MSFT_FY2024_item7_000` (MSFT FY2024 item7) score=0.670
  4. `MSFT_FY2020_item7_000` (MSFT FY2020 item7) score=0.667
  5. `MSFT_FY2023_item7_000` (MSFT FY2023 item7) score=0.665

**Answer:**

> $198,270 million

---

### L2 (years=[2022])

Retrieved chunks:

  1. `MSFT_FY2023_item8_086` (MSFT FY2023 item8) score=0.689
  2. `MSFT_FY2022_item7_000` (MSFT FY2022 item7) score=0.670
  3. `MSFT_FY2024_item7_000` (MSFT FY2024 item7) score=0.670
  4. `MSFT_FY2020_item7_000` (MSFT FY2020 item7) score=0.667 · **KG-expanded**
  5. `MSFT_FY2023_item7_000` (MSFT FY2023 item7) score=0.665 · **KG-expanded**

**Answer:**

> $198,270 million

---

