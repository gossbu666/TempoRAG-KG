# KG Triple Filter Report

**Input:** `data/kg/full/triples.jsonl`
**Output:** `data/kg/filtered/triples.jsonl`

## Headline

- Total input triples: **60436**
- Kept: **57718** (95.5%)
- Dropped: **2718** (4.5%)

## Drop reasons

| Reason | Count | % of input |
|---|---:|---:|
| tooshort_sp | 1426 | 2.36% |
| bool_literal_object | 1139 | 1.88% |
| yesno_literal_object | 92 | 0.15% |
| long_object | 52 | 0.09% |
| self_loop | 9 | 0.01% |

## Kept triples — by ticker

| Ticker | Kept |
|---|---:|
| AAPL | 5573 |
| ADBE | 3897 |
| AMZN | 6096 |
| CSCO | 5860 |
| GOOGL | 6695 |
| INTC | 5814 |
| META | 7426 |
| MSFT | 8044 |
| NVDA | 3608 |
| ORCL | 4705 |

## Kept triples — by item

| Item | Kept |
|---|---:|
| 1 | 5618 |
| 1A | 7002 |
| 7 | 10765 |
| 7A | 971 |
| 8 | 33362 |

## Kept triples — by fiscal year

| FY | Kept |
|---|---:|
| 2019 | 5224 |
| 2020 | 5430 |
| 2021 | 5028 |
| 2022 | 13021 |
| 2023 | 14376 |
| 2024 | 14639 |

## Kept triples — by temporal type

| Temporal type | Kept | % of kept |
|---|---:|---:|
| explicit | 51048 | 88.4% |
| unknown | 6246 | 10.8% |
| forward_looking | 424 | 0.7% |
