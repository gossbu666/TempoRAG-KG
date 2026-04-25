"""Run the 6 Streamlit demo experiments programmatically and dump a report.

Mirrors the user-facing Streamlit flow (`app/streamlit_app.py`) without the
browser: same retrieval functions, same answer model. Output lands in
`docs/streamlit_experiment_results.md` so the user can include the
findings directly in slides / video script / report.

Each experiment specifies:
  - question
  - year filter
  - one or more conditions to compare (L0/L1/L2/L3)
  - expected behavior

For each (experiment, condition), we record:
  - top-k chunk_ids + scores + KG-expanded flags (when applicable)
  - the model's answer (gpt-4o-mini, temp 0)
  - F1 vs gold (if the question matches a labeled QA record)
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src.answer import (
    MODEL_REGISTRY,
    answer_question,
    build_client,
    load_answer_template,
)
from src.cache import Cache
from src.eval import f1_token
from src.retrieval import (
    OpenAIEmbeddingClient,
    load_index,
    load_kg_index,
    retrieve,
    retrieve_kg2rag,
    retrieve_temporag_kg,
    retrieve_with_year_filter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
INDEX_DIR = REPO_ROOT / "data" / "embeddings" / "chunks"
TRIPLES_PATH = REPO_ROOT / "data" / "kg" / "filtered" / "triples.jsonl"
ANSWER_PROMPT = REPO_ROOT / "prompts" / "answer_v1.txt"
OUT = REPO_ROOT / "docs" / "streamlit_experiment_results.md"

ANSWER_MODEL = "gpt-4o-mini"  # consistent across experiments

EXPERIMENTS = [
    {
        "id": "E1",
        "title": "Smoke — easy single-hop (all conditions converge)",
        "question": "What was Apple's revenue in fiscal 2022?",
        "years_by_cond": {"L0": [], "L1": [2022], "L2": [], "L3": [2022]},
        "conditions": ["L0", "L1", "L2", "L3"],
        "hypothesis": "All four conditions return the same chunks and answer "
                      "because the gold-bearing chunk dominates cosine.",
    },
    {
        "id": "E2",
        "title": "L1 year-mask actually filters (no year in question)",
        "question": "What was Apple's revenue?",
        "years_by_cond": {"L0": [], "L1": [2020]},
        "conditions": ["L0", "L1"],
        "hypothesis": "L0 returns recent chunks (FY2023/FY2024 bias). "
                      "L1 with year=[2020] forces FY2020 chunk → answer is "
                      "FY2020 revenue (~$274.5B), not the latest.",
    },
    {
        "id": "E3",
        "title": "L2 KG²RAG graph walk vs L0 cosine",
        "question": "Compare Apple's Services revenue to Microsoft's cloud "
                    "revenue for fiscal year 2022",
        "years_by_cond": {"L0": [], "L2": []},
        "conditions": ["L0", "L2"],
        "hypothesis": "L0 may favor a single ticker; L2 entity-walk should "
                      "include chunks from the second ticker via shared "
                      "entities (KG-expanded flag visible).",
    },
    {
        "id": "E4",
        "title": "L3 inter-year (correctly scoped year filter)",
        "question": "How did Amazon's AWS operating income evolve from 2020 to 2023?",
        "years_by_cond": {"L3": [2020, 2021, 2022, 2023]},
        "conditions": ["L3"],
        "hypothesis": "With a 4-year filter, L3 keeps triples whose validity "
                      "intersects 2020-2023 → answer covers all 4 years.",
    },
    {
        "id": "E5",
        "title": "A4 IDK-when-answerable demonstration",
        "question": "What risk factors did Meta disclose regarding EU "
                    "regulations in 2023?",
        "years_by_cond": {"L1": [2023]},
        "conditions": ["L1"],
        "hypothesis": "Even with the right Meta FY2023 Item 1A chunks "
                      "retrieved, gpt-4o-mini abstains with 'I don't know'. "
                      "Demonstrates the 41.8% A4 finding.",
    },
    {
        "id": "E6",
        "title": "L2 regression — KG²RAG ≈ L0 on easy intra query",
        "question": "What was Microsoft's revenue in fiscal 2022?",
        "years_by_cond": {"L0": [2022], "L2": [2022]},
        "conditions": ["L0", "L2"],
        "hypothesis": "Same chunks, same answer — L2 graph walk doesn't add "
                      "value when cosine seeds already span the answer.",
    },
]


def _qa_lookup() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in ("data/qa/home_grown.jsonl", "data/qa/multihop_filtered.jsonl"):
        p = REPO_ROOT / src
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r["question"].strip().lower()] = r
    return out


def _retrieve(condition, question, years, index, kg, embed_client, qcache):
    if condition == "L0":
        return retrieve(question, index, embed_client, qcache, k=5)
    if condition == "L1":
        return retrieve_with_year_filter(question, years, index, embed_client,
                                         qcache, k=5)
    if condition == "L2":
        return retrieve_kg2rag(question, index, kg, embed_client, qcache,
                               k=5, seed_k=3)
    if condition == "L3":
        return retrieve_temporag_kg(question, years, index, kg, embed_client,
                                    qcache, k=5, seed_k=3)
    raise ValueError(condition)


def _format_chunks(chunks, seed_cids=None) -> str:
    out = []
    for i, c in enumerate(chunks, 1):
        kg_tag = ""
        if seed_cids is not None and c["chunk_id"] not in seed_cids:
            kg_tag = " · **KG-expanded**"
        out.append(f"  {i}. `{c['chunk_id']}` ({c.get('ticker','?')} "
                   f"FY{c.get('fy','?')} item{c.get('item','?')}) "
                   f"score={c['retrieval_score']:.3f}{kg_tag}")
    return "\n".join(out)


def main() -> None:
    load_dotenv()
    print("Loading indices ...")
    index = load_index(INDEX_DIR, CHUNKS_PATH)
    kg = load_kg_index(TRIPLES_PATH)
    embed_client = OpenAIEmbeddingClient()
    qcache = Cache(REPO_ROOT / "data" / "cache" / "query_embed")
    acache = Cache(REPO_ROOT / "data" / "cache" / "answer")
    template = load_answer_template(ANSWER_PROMPT)
    answer_client = build_client(ANSWER_MODEL)
    _, model_id = MODEL_REGISTRY[ANSWER_MODEL]
    qa_by_question = _qa_lookup()
    print(f"Loaded: {len(index.chunk_ids)} chunks, "
          f"{len(kg.chunk_to_triples)} chunks w/ triples\n")

    sections: list[str] = [
        "# Streamlit demo experiments — programmatic run",
        "",
        f"Run via `scripts/run_demo_experiments.py` on the same retrieval + "
        f"answer pipeline as `app/streamlit_app.py`. Answer model: "
        f"`{model_id}` at temperature 0. Cache shared with the eval pipeline, "
        f"so re-runs are free.",
        "",
        "Each experiment lists:",
        "- the question (verbatim)",
        "- year filter for temporal conditions",
        "- per-condition retrieved chunks (top-5 with cosine scores; "
        "KG-expanded chunks tagged for L2/L3)",
        "- the model's answer",
        "- gold + Token-F1 if the question matches a labeled QA record",
        "",
        "---",
        "",
    ]

    for exp in EXPERIMENTS:
        print(f"=== {exp['id']}: {exp['title']} ===")
        sections += [
            f"## {exp['id']} — {exp['title']}",
            "",
            f"**Question:** {exp['question']}",
            "",
            f"**Hypothesis:** {exp['hypothesis']}",
            "",
        ]

        gold_rec = qa_by_question.get(exp["question"].strip().lower())

        # Compute seed_cids per L2/L3 to mark KG-expanded chunks.
        seed_cids: set[str] = set()
        if any(c in ("L2", "L3") for c in exp["conditions"]):
            seeds = retrieve(exp["question"], index, embed_client, qcache, k=3)
            seed_cids = {s["chunk_id"] for s in seeds}

        for cond in exp["conditions"]:
            years = exp["years_by_cond"].get(cond, [])
            chunks = _retrieve(cond, exp["question"], years, index, kg,
                               embed_client, qcache)
            ans = answer_question(exp["question"], chunks, answer_client, acache,
                                  template=template, model=model_id)
            sec_seed = seed_cids if cond in ("L2", "L3") else None

            sections += [
                f"### {cond} (years={years if years else '—'})",
                "",
                "Retrieved chunks:",
                "",
                _format_chunks(chunks, seed_cids=sec_seed),
                "",
                "**Answer:**",
                "",
                f"> {ans['answer']}",
                "",
            ]
            if gold_rec:
                gold = gold_rec["answer"]
                if isinstance(gold, list):
                    gold_str = " | ".join(str(g) for g in gold)
                else:
                    gold_str = str(gold)
                f1 = f1_token(ans["answer"], gold)
                sections += [
                    f"_gold (from labeled set):_ {gold_str}",
                    "",
                    f"_Token-F1 vs gold: **{f1:.3f}**_",
                    "",
                ]
            sections.append("---")
            sections.append("")
            print(f"  {cond}: {ans['answer'][:80]}")

        sections.append("")

    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote: {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
