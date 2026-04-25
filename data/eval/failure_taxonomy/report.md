# Failure Taxonomy — Aggregate Report

Total classified predictions: **3607**
Non-failures (NF): **662** (18.4%)
Unlabeled remaining: **0** (expected 0 after Stage 2)

## Headline counts by category

| Code | Count | % |
|------|------:|---:|
| A1 | 864 | 24.0% |
| A2 | 319 | 8.8% |
| A3 | 0 | 0.0% |
| A4 | 1508 | 41.8% |
| A5 | 60 | 1.7% |
| B1 | 0 | 0.0% |
| B2 | 25 | 0.7% |
| B3 | 118 | 3.3% |
| B4 | 0 | 0.0% |
| B5 | 51 | 1.4% |
| NF | 662 | 18.4% |
| unlabeled | 0 | 0.0% |

## Tables

- [by_model.csv](by_model.csv)
- [by_condition.csv](by_condition.csv)
- [by_scope.csv](by_scope.csv)
- [by_hop.csv](by_hop.csv)
- [examples.md](examples.md)

## Reliability

**LLM-vs-LLM agreement** between Stage 2 (gpt-4o-mini, production classifier) and a second judge (gpt-4o) on a stratified sample of 20 rows: observed agreement **60.0\%**, Cohen's $\kappa$ = **0.200** (slight).

This is _not_ formal human inter-rater reliability. We frame it as cross-LLM consistency evidence; the size of $\kappa$ reflects how reproducible the Stage 2 labels are when a larger model judges the same prompt.

Sample file: [`kappa_llm_sample.jsonl`](kappa_llm_sample.jsonl).

