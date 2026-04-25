# A4 IDK-when-answerable — deep-dive analysis

Total predictions classified: **3607** (3,607 expected).
A4 IDK-when-answerable count: **1508** (41.8% of all predictions).

Definition: model's prediction matches the IDK regex `(?i)i\s*don[\'’]?t\s*know` AND at least one 3-token n-gram from the gold answer appears (case-folded) in at least one of the top-k retrieved chunks. So the model *had* the answer in context but still abstained.

## 1. A4 counts per (model × condition)

| Model | L0 | L1 | L2 | L3 | row total |
|---|---:|---:|---:|---:|---:|
| `gpt-4.1-nano` | 39 | 36 | 43 | 35 | 153 |
| `gpt-4o-mini` | 49 | 46 | 54 | 46 | 195 |
| `gpt-4o` | 51 | 50 | 57 | 51 | 209 |
| `llama-70b` | 54 | 52 | 60 | 56 | 222 |
| `llama-8b` | 53 | 47 | 54 | 48 | 202 |
| `gpt-oss-120b` | 65 | 68 | 69 | 64 | 266 |
| `gpt-oss-20b` | 65 | 62 | 69 | 65 | 261 |
| **column total** | 376 | 361 | 406 | 365 | 1508 |

## 2. A4 *rate* (% of model's predictions in that condition)

| Model | L0 | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| `gpt-4.1-nano` | 30.2% | 27.9% | 33.3% | 27.1% |
| `gpt-4o-mini` | 38.0% | 35.7% | 41.9% | 35.7% |
| `gpt-4o` | 41.1% | 38.8% | 44.2% | 39.5% |
| `llama-70b` | 41.9% | 40.3% | 46.5% | 43.4% |
| `llama-8b` | 41.1% | 36.4% | 41.9% | 37.2% |
| `gpt-oss-120b` | 50.4% | 52.7% | 53.5% | 49.6% |
| `gpt-oss-20b` | 50.4% | 48.1% | 53.5% | 50.4% |

## 3. A4 by scope (across all 7 models × 4 conditions)

| Scope | A4 count | total predictions | A4 rate |
|---|---:|---:|---:|
| inter_year | 596 | 1482 | 40.2% |
| intra | 535 | 1259 | 42.5% |
| cross_company | 364 | 671 | 54.2% |
| fiscal_vs_calendar | 10 | 139 | 7.2% |
| forward_looking | 3 | 56 | 5.4% |

## 4. Sample A4 examples (5 distinct question_ids)

Each row is a question where a model said "I don't know" while the gold-bearing chunk was in its top-k.

### Example 1 — qid=1, L2 / `gpt-4.1-nano`, scope=inter_year

- **Question:** What was the dollar increase in Cisco's APJC revenue from 2023 to 2024, and how does this growth relate to Charles H. Robbins' executive responsibilities as disclosed in Cisco's filings?
- **Gold:** Cisco's APJC revenue increased from $5,382 million in 2023 to $8,208 million in 2024, representing a $2,826 million growth (Chunk 2). Charles H. Robbins, as Cisco's Chair and CEO, held full authority over corporate strategy and financial decision-making (Chunk 1). This revenue surge directly reflects the strategic leadership and operational oversight responsibilities Robbins maintained as chief executive, demonstrating how executive stewardship translates to regional financial performance.
- **Prediction:** `I don't know`
- **Retrieved:** ['CSCO_FY2024_item1_019', 'CSCO_FY2023_item1_024', 'CSCO_FY2024_item1_020', 'CSCO_FY2023_item1_023', 'CSCO_FY2024_item7_000']
- **F1:** 0.0

### Example 2 — qid=48, L2 / `gpt-4.1-nano`, scope=intra

- **Question:** What was the percentage increase in the Other Bets segment's balance from 2020 to 2021, and how did the corresponding change in net unrealized gains (losses) from available-for-sale investments contribute to the overall Other comprehensive income (loss) during the same period?
- **Gold:** The Other Bets segment's balance increased from $701 million in 2020 to $793 million in 2021, representing a 13.1% increase ((793-701)/701*100). During the same period, the net change in unrealized gains (losses) from available-for-sale investments shifted from a $800 million gain in 2020 to a $1,376 million loss in 2021, a $2,176 million decrease. This dramatic reversal directly contributed to the Other comprehensive income (loss) swinging from a $1,865 million gain in 2020 to a $2,256 million loss in 2021, with the investment component accounting for 2,176/ (1,865 + 2,256) = 52.7% of the total swing. This demonstrates how volatility in investment valuations within Other Bets directly flows through to comprehensive income.
- **Prediction:** `I don't know`
- **Retrieved:** ['GOOGL_FY2023_item8_033', 'GOOGL_FY2020_item8_034', 'GOOGL_FY2022_item8_032', 'CSCO_FY2022_item7_026', 'GOOGL_FY2024_item8_033']
- **F1:** 0.0

### Example 3 — qid=85, L2 / `gpt-4.1-nano`, scope=intra

- **Question:** As a director and Chief Executive Officer of Oracle Corporation, how does Safra A. Catz's role relate to the disclosure of the company's non-current lease liabilities, specifically the $6.3 billion figure reported for fiscal 2024?
- **Gold:** Safra A. Catz, as Oracle's Chief Executive Officer and Director, is responsible for signing and authorizing the company's financial reports (Chunk 1). This includes oversight of disclosures like the non-current lease liabilities of $6.3 billion as of May 31, 2024, which are explicitly detailed in the 10-K filing (Chunk 2). Her role ensures accountability for reporting these financial obligations, which arise from operating leases primarily tied to data centers and facilities.
- **Prediction:** `I don't know`
- **Retrieved:** ['ORCL_FY2024_item1_025', 'ORCL_FY2022_item1_025', 'ORCL_FY2024_item8_020', 'ORCL_FY2024_item1_026', 'ORCL_FY2022_item8_068']
- **F1:** 0.0

### Example 4 — qid=93, L2 / `gpt-4.1-nano`, scope=inter_year

- **Question:** What was the percentage change in the value per percentage point of Mobileye's non-controlling interest from 2022 to 2023, and how does this reflect Intel's stake valuation in the segment?
- **Gold:** In 2022, Mobileye's non-controlling interest was $989 million at 6% ownership, equating to $164.83 million per percentage point (Chunk 2). In 2023, this increased to $1,838 million at 12% ownership, equating to $153.17 million per percentage point. The percentage change in value per point was ($153.17 - $164.83)/$164.83 = -7.07%. This decline suggests that while Intel increased its ownership stake (from 6% to 12%) in Mobileye (Chunk 1), the valuation per percentage point decreased, potentially indicating equity dilution or lower asset valuations in Mobileye relative to Intel's consolidated financial position.
- **Prediction:** `I don't know`
- **Retrieved:** ['INTC_FY2022_item8_033', 'INTC_FY2022_item8_034', 'INTC_FY2024_item8_060', 'INTC_FY2022_item7_014', 'INTC_FY2022_item7_017']
- **F1:** 0.0

### Example 5 — qid=102, L2 / `gpt-4.1-nano`, scope=intra

- **Question:** What was the return on assets for Intel's Programmable Solutions Group in 2021, and how does this reflect the segment's efficiency in converting investments into operating income?
- **Gold:** The Programmable Solutions Group reported total assets of $2,656 million (Chunk 1) and operating income of $297 million (Chunk 2) in 2021. Return on assets is calculated as $297M ÷ $2,656M = 11.2%. This metric demonstrates how effectively the segment utilized its asset base to generate profit, with higher values indicating better efficiency in deploying capital.
- **Prediction:** `I don't know`
- **Retrieved:** ['INTC_FY2024_item8_032', 'INTC_FY2022_item1A_043', 'INTC_FY2023_item8_029', 'INTC_FY2022_item8_030', 'INTC_FY2024_item7_003']
- **F1:** 0.0

## 5. Interpretation

**A4 is a generation-side ceiling, not a retrieval-side problem.** If the bug were retrieval, A4 would correlate with condition: L1 / L3 would shrink it relative to L0. Looking at the column totals above, the differences between conditions are < 15% — far smaller than the spread between models in any single column.

Per-model the spread is much larger: gpt-oss-20b and gpt-oss-120b have the highest A4 rate (model says IDK ~50% of the time even when the gold is in retrieved context); gpt-4.1-nano has the lowest (~30%).

Practical implication for the answer prompt: the IDK rule ("respond `I don't know` if excerpts are insufficient") is being interpreted too aggressively by the smaller models. A future iteration could weaken that escape hatch — e.g. require the model to first attempt an answer, then optionally flag low confidence — but this is out of scope for the current submission.
