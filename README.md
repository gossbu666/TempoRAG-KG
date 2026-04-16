# TempoRAG-KG

Temporally-aware Knowledge Graph-augmented Retrieval for multi-hop question answering.
Course project for AIT NLU, Spring 2026.

Extends KG²RAG (NAACL 2025) by attaching explicit `[valid_from, valid_to]` validity
intervals to every KG triple, enabling deterministic temporal filtering during
retrieval on time-sensitive questions.

## Status

**Phase 1 — de-risking / infrastructure setup.** Proposal submitted (March 2026);
revised plan in response to TA feedback; currently preparing pilot extraction and
evaluation harness before full KG build.

## Team

- Supanut Kompayak (st126055) — lead
- Aphisit Jaemyaem (st126130)
- Dechathon Niamsa-ard (st126235)
- Kaung Hein Htet (st126477)

## Repository layout

```
.
├── data/              # datasets (raw files gitignored; sampled IDs committed)
│   └── samples/       # deterministic sampled question IDs
├── src/               # pipeline modules
├── prompts/           # LLM prompt templates
├── tests/             # unit tests
├── scripts/           # one-off shell scripts (data download, etc.)
├── docs/              # protocols, pilot reports
├── results/           # EDA output, experiment results
├── proposal/          # LaTeX proposal + PDFs
└── slides/            # presentation decks
```

## Setup

```bash
# 1. Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env
# then edit .env with real API keys

# 3. Datasets
bash scripts/download_data.sh
```

## Research questions

- **RQ1** — What types of temporal failures occur in KG-RAG, and how are they distributed?
- **RQ2** — Does explicit triple-level temporal tagging improve F1 on temporal subsets vs KG²RAG?
- **RQ3** — How accurately can LLMs extract `[valid_from, valid_to]` from unstructured text?
- **RQ4** — Does the benefit of temporal filtering scale inversely with generator capability?

## Budget discipline

Student-funded, hard cap $20. All paid API responses are disk-cached in `cache/`
(keyed by `sha256(model + prompt)`). Cache layer must be in place before any paid
API call.

## License

Course project — internal use.
