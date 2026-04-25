# Synth pool auto-vet report

Judge model: `gpt-4o-2024-11-20` (temperature 0.0).

Input: `data/qa/synth_pool_verified.jsonl` (128 rows).

**Accepted: 4 / 128**  (3.1%).
Rejected: 124.

## Rejection breakdown by failed axis

| Axis | Rejected count |
|---|---:|
| answer_correct | 94 |
| prior_hop_unverified | 93 |
| hop_correct | 88 |
| scope_correct | 7 |

Output files: `data/qa/synth_pool_accepted.jsonl`, `data/qa/synth_pool_rejected.jsonl`.
