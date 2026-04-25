# FinReflectKG Subset Report

**Source:** HuggingFace `domyn/FinReflectKG` (streaming)
**Filter:** ticker ∈ ['AAPL', 'ADBE', 'AMZN', 'CSCO', 'GOOGL', 'INTC', 'META', 'MSFT', 'NVDA', 'ORCL'], year ∈ [2022, 2023, 2024]
**Output:** `data/finreflectkg/triples.jsonl`
**Total triples kept:** 74,979

## Triples per ticker

| Ticker | Triples |
|---|---:|
| AAPL | 4,676 |
| ADBE | 7,504 |
| AMZN | 6,484 |
| CSCO | 9,678 |
| GOOGL | 7,131 |
| INTC | 9,264 |
| META | 6,135 |
| MSFT | 8,589 |
| NVDA | 7,109 |
| ORCL | 8,409 |

## Triples per ticker × year

| Ticker | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| AAPL | 1,504 | 1,571 | 1,601 |
| ADBE | 2,496 | 2,454 | 2,554 |
| AMZN | 2,022 | 2,207 | 2,255 |
| CSCO | 3,126 | 3,356 | 3,196 |
| GOOGL | 2,377 | 2,396 | 2,358 |
| INTC | 3,110 | 2,859 | 3,295 |
| META | 0 | 3,056 | 3,079 |
| MSFT | 2,587 | 2,983 | 3,019 |
| NVDA | 2,128 | 2,579 | 2,402 |
| ORCL | 2,791 | 2,880 | 2,738 |

## Extraction type breakdown

FinReflectKG's `extraction_type` field distinguishes triples whose dates were LLM-extracted (`explicit`, `relative`) from those that default to the filing year (`default`). This matters for our baseline comparison — TempoRAG-KG's value proposition is precisely that our explicit-extracted intervals outperform default-filing-year fallback.

| extraction_type | Triples |
|---|---:|
| default | 57,401 |
| extracted | 17,255 |
| dropped | 255 |
| merged | 9 |
| default_start_timestamp | 5 |
| due to increased regulatory scrutiny | 4 |
| RAW_MATERIAL | 2 |
| default_end_timestamp | 2 |
| financial disclosure | 2 |
| consolidated | 2 |
| Marketing Campaigns | 1 |
| Improved User Experience | 1 |
| due to Tax Rate Changes | 1 |
| due to recent acquisitions | 1 |
| due to increased employee incentives | 1 |
| INTC depends_on SolarWinds for IT Management | 1 |
| Competition | 1 |
| June 2023 | 1 |
| December 2022 | 1 |
| Due To Increased Costs | 1 |
| due_to_REGULATORY_REQUIREMENT | 1 |
| due to competition in the social media market | 1 |
| due to competition in the advertising market | 1 |
| due to increased competition | 1 |
| due to changes in platform algorithms | 1 |
| due to regulatory changes | 1 |
| due to increased operating costs | 1 |
| due to data breaches | 1 |
| due to decreased ad revenue | 1 |
| redundant | 1 |
| Financial Disclosure | 1 |
| due_to Privacy Concerns | 1 |
| due_to Competitor Offers | 1 |
| due_to API Changes | 1 |
| due_to Ad Blockers | 1 |
| Supply Chain Disruptions | 1 |
| Due_To Supply Chain Disruptions | 1 |
| Due_To Currency Fluctuations | 1 |
| due to Supply Chain Disruptions | 1 |
| due to Product Launch | 1 |
| Market Decline | 1 |
| removed | 1 |
| due_to Economic Downturn | 1 |
| due_to New Product Launch | 1 |
| combined | 1 |
| due_to Economic_Downturn | 1 |
| due_to Regulatory_Changes | 1 |
| Foreign Currency Fluctuations | 1 |
| New Product Launch | 1 |
| due to increased competition and higher marketing costs | 1 |
| due to increased competition and higher operating costs | 1 |
| default_context | 1 |

**Default rate:** 76.6% of triples use the filing-year default.
