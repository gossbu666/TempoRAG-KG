# Per-question difficulty analysis

Each question's difficulty score = mean Token-F1 across all 7 models $\times$ 4 retrieval conditions = 28 prediction cells. Lower = harder.

Total questions analysed: **129**.

## 1. Top-10 hardest questions (lowest mean F1)

| Rank | QID | Avg F1 | Scope | Hop | Ticker(s) |
|---:|---|---:|---|---:|---|
| 1 | 1 | 0.000 | inter_year | 2 | CSCO |
| 2 | 51 | 0.000 | intra | 2 | GOOGL |
| 3 | 85 | 0.000 | intra | 2 | ORCL |
| 4 | 102 | 0.000 | intra | 2 | INTC |
| 5 | 110 | 0.000 | intra | 2 | CSCO |
| 6 | 173 | 0.000 | intra | 3 | INTC |
| 7 | 276 | 0.000 | inter_year | 3 | INTC |
| 8 | 324 | 0.000 | intra | 2 | META |
| 9 | 329 | 0.000 | inter_year | 3 | GOOGL |
| 10 | 336 | 0.000 | inter_year | 3 | GOOGL |

### Pattern in hardest 10
- **Scope distribution:** {'inter_year': 4, 'intra': 6}
- **Hop distribution:** {2: 6, 3: 4}
- **Top tickers:** {'GOOGL': 3, 'INTC': 3, 'CSCO': 2, 'ORCL': 1, 'META': 1}

## 2. Top-10 easiest questions (highest mean F1)

| Rank | QID | Avg F1 | Scope | Hop | Ticker(s) |
|---:|---|---:|---|---:|---|
| 1 | H019 | 0.719 | inter_year | 2 | AAPL |
| 2 | H024 | 0.716 | inter_year | 2 | AMZN |
| 3 | H029 | 0.694 | inter_year | 2 | AAPL |
| 4 | H020 | 0.659 | inter_year | 2 | AAPL |
| 5 | H030 | 0.578 | inter_year | 2 | AAPL |
| 6 | H003 | 0.518 | cross_company | 2 | AMZN,MSFT |
| 7 | H007 | 0.495 | cross_company | 2 | AAPL,MSFT |
| 8 | H038 | 0.470 | intra | 2 | AMZN |
| 9 | 537 | 0.420 | inter_year | 2 | CSCO |
| 10 | 82 | 0.417 | inter_year | 2 | INTC |

### Pattern in easiest 10
- **Scope distribution:** {'inter_year': 7, 'cross_company': 2, 'intra': 1}
- **Hop distribution:** {2: 10}
- **Top tickers:** {'AAPL': 5, 'AMZN': 3, 'MSFT': 2, 'CSCO': 1, 'INTC': 1}

## 3. Worked examples

### Three hardest

### qid 1 — avg F1 0.000  (L0=0.00  L1=0.00  L2=0.00  L3=0.00)
- **Scope / hop:** inter_year / 2
- **Q:** What was the dollar increase in Cisco's APJC revenue from 2023 to 2024, and how does this growth relate to Charles H. Robbins' executive responsibilities as disclosed in Cisco's filings?
- **Gold:** Cisco's APJC revenue increased from $5,382 million in 2023 to $8,208 million in 2024, representing a $2,826 million growth (Chunk 2). Charles H. Robbins, as Cisco's Chair and CEO, held full authority 

### qid 51 — avg F1 0.000  (L0=0.00  L1=0.00  L2=0.00  L3=0.00)
- **Scope / hop:** intra / 2
- **Q:** Calculate the percentage change in Google's interest income from 2020 to 2021 and compare it with the percentage change in the Google Services Segment's balance during the same period.
- **Gold:** Google's interest income decreased from $1,865M in 2020 to $1,499M in 2021, representing a 19.6% decline ((1,499 - 1,865)/1,865 = -0.196). Meanwhile, the Google Services Segment's balance increased fr

### qid 85 — avg F1 0.000  (L0=0.00  L1=0.00  L2=0.00  L3=0.00)
- **Scope / hop:** intra / 2
- **Q:** As a director and Chief Executive Officer of Oracle Corporation, how does Safra A. Catz's role relate to the disclosure of the company's non-current lease liabilities, specifically the $6.3 billion figure reported for fiscal 2024?
- **Gold:** Safra A. Catz, as Oracle's Chief Executive Officer and Director, is responsible for signing and authorizing the company's financial reports (Chunk 1). This includes oversight of disclosures like the n

### Three easiest

### qid H019 — avg F1 0.719  (L0=0.69  L1=0.72  L2=0.73  L3=0.73)
- **Scope / hop:** inter_year / 2
- **Q:** How did Apple's total net sales change from fiscal 2019 to fiscal 2021?
- **Gold:** Apple's total net sales grew from $260,174 million in fiscal 2019 to $365,817 million in fiscal 2021, an increase of $105,643 million or approximately 40.6%.

### qid H024 — avg F1 0.716  (L0=0.70  L1=0.75  L2=0.71  L3=0.71)
- **Scope / hop:** inter_year / 2
- **Q:** How did Amazon's total net sales grow from fiscal 2019 to fiscal 2021?
- **Gold:** Amazon's total net sales grew from $280,522 million in fiscal 2019 to $469,822 million in fiscal 2021, an increase of $189,300 million or approximately 67.5% — reflecting substantial pandemic-era dema

### qid H029 — avg F1 0.694  (L0=0.68  L1=0.74  L2=0.68  L3=0.68)
- **Scope / hop:** inter_year / 2
- **Q:** How did Apple's Services revenue change from fiscal 2020 to fiscal 2021?
- **Gold:** Apple's Services revenue grew from $53,768 million in fiscal 2020 to $68,425 million in fiscal 2021, an increase of $14,657 million or approximately 27.3%.

## 4. Difficulty distribution

- Mean F1 across all 129 questions: **0.175**
- Median F1: **0.135**
- Std dev:   **0.169**
- Questions with avg F1 == 0 (every cell failed): **20**
- Questions with avg F1 \geq 0.8 (consistently solved): **0**
