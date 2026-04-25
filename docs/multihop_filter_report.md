# FinReflectKG-MultiHop Filter Report

**Source:** `data/multihop_qa/final_master_dataset.json` (555 QA pairs)
**Filter:** all hops' source filings in ['AAPL', 'ADBE', 'AMZN', 'CSCO', 'GOOGL', 'INTC', 'META', 'MSFT', 'NVDA', 'ORCL']
**Output:** `data/qa/multihop_filtered.jsonl`
**Kept:** 79 / 555

## By scope

| Scope | Kept |
|---|---:|
| intra | 35 |
| inter_year | 38 |
| cross_company | 6 |

## By ticker (kept questions touching each ticker)

| Ticker | Qs touching |
|---|---:|
| AAPL | 3 |
| ADBE | 1 |
| AMZN | 0 |
| CSCO | 15 |
| GOOGL | 12 |
| INTC | 24 |
| META | 6 |
| MSFT | 5 |
| NVDA | 6 |
| ORCL | 13 |

## Notes

- Strict filter: every hop in a kept question references a 10-K whose ticker is in our target set. A question that mentions a ticker outside the set (even in one hop) is dropped, because our KG has no entries for that company and the generator would be asked to answer from facts we don't carry.
- `scope` field: re-mapped from the paper's `document_relationship`. Used as the stratification key by `src.eval.aggregate_by_scope`.
- The `cross_company` bucket was 0 under the Mag5 filter; the 10-ticker expansion (adding CSCO/ORCL/INTC/NVDA/ADBE) unlocks this scope and lets us measure the temporal-KG lift on its most-distinctive regime.