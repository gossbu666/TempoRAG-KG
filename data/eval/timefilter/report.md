# Timefilter RAG Evaluation

**Coverage** = fraction of rows where the model attempted an answer (non-IDK, non-empty). **F1@answered** = token-F1 on the answered subset — factors out retrieval-coverage failures from answer-quality failures.

| Model | Provider | F1 | F1 CI | Coverage | F1@answered | n_answered |
|---|---|---:|---|---:|---:|---:|
| `gpt-4.1-nano` | openai | 0.258 | [0.221, 0.296] | 70.5% | 0.366 [0.333, 0.398] | 91 |
| `gpt-4o-mini` | openai | 0.248 | [0.206, 0.290] | 62.8% | 0.395 [0.357, 0.435] | 81 |
| `gpt-4o` | openai | 0.221 | [0.184, 0.264] | 56.6% | 0.391 [0.348, 0.432] | 73 |
| `llama-8b` | openrouter | 0.180 | [0.147, 0.216] | 57.4% | 0.314 [0.276, 0.355] | 74 |
| `llama-70b` | openrouter | 0.179 | [0.144, 0.215] | 53.5% | 0.335 [0.294, 0.373] | 69 |
| `gpt-oss-20b` | openrouter | 0.145 | [0.110, 0.184] | 42.6% | 0.336 [0.279, 0.391] | 55 |
| `gpt-oss-120b` | openrouter | 0.138 | [0.103, 0.181] | 38.8% | 0.356 [0.296, 0.418] | 50 |

_Re-aggregated: 2026-04-20 03:50:21_
