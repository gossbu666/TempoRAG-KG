# Vanilla RAG Evaluation

**Coverage** = fraction of rows where the model attempted an answer (non-IDK, non-empty). **F1@answered** = token-F1 on the answered subset — factors out retrieval-coverage failures from answer-quality failures.

| Model | Provider | F1 | F1 CI | Coverage | F1@answered | n_answered |
|---|---|---:|---|---:|---:|---:|
| `gpt-4o-mini` | openai | 0.230 | [0.193, 0.274] | 58.1% | 0.396 [0.356, 0.437] | 75 |
| `gpt-4.1-nano` | openai | 0.229 | [0.194, 0.270] | 63.6% | 0.360 [0.325, 0.394] | 82 |
| `gpt-4o` | openai | 0.183 | [0.141, 0.223] | 47.6% | 0.385 [0.337, 0.433] | 59 |
| `llama-70b` | openrouter | 0.145 | [0.112, 0.181] | 43.4% | 0.333 [0.291, 0.374] | 56 |
| `llama-8b` | openrouter | 0.145 | [0.114, 0.180] | 48.1% | 0.301 [0.258, 0.339] | 62 |
| `gpt-oss-120b` | openrouter | 0.131 | [0.094, 0.172] | 35.7% | 0.366 [0.306, 0.435] | 46 |
| `gpt-oss-20b` | openrouter | 0.120 | [0.088, 0.159] | 37.2% | 0.320 [0.255, 0.386] | 48 |

_Re-aggregated: 2026-04-20 03:37:05_
