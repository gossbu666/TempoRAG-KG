"""L2 KG²RAG and L3 TempoRAG-KG evaluation sweep (7 models × 129 Qs).

Parallels `run_vanilla_eval.py` and `run_timefilter_eval.py` but uses the
filtered KG (`data/kg/filtered/triples.jsonl`) to expand retrieval:

- `--condition kg2rag`     → L2 : cosine seeds + 1-hop KG expansion, no temporal
- `--condition temporag_kg`→ L3 : L2 + triple-level temporal filter on Q.years

Output:
  data/eval/kg2rag/<model>/predictions.jsonl + summary.json
  data/eval/temporag/<model>/predictions.jsonl + summary.json
  report.md at the condition root aggregates across models.

Answer-call cache is keyed on (model, rendered_prompt, temperature), so a
re-run with the same retrieval + QA hits cache row-by-row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.answer import (
    MODEL_REGISTRY,
    answer_question,
    build_client,
    load_answer_template,
)
from src.cache import Cache
from src.eval import aggregate, em, f1_token
from src.retrieval import (
    ChunkIndex,
    KGIndex,
    OpenAIEmbeddingClient,
    load_index,
    load_kg_index,
    retrieve_kg2rag,
    retrieve_temporag_kg,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
INDEX_DIR = REPO_ROOT / "data" / "embeddings" / "chunks"
TRIPLES_PATH = REPO_ROOT / "data" / "kg" / "filtered" / "triples.jsonl"
ANSWER_CACHE = REPO_ROOT / "data" / "cache" / "answer"
QUERY_EMBED_CACHE = REPO_ROOT / "data" / "cache" / "query_embed"
ANSWER_PROMPT = REPO_ROOT / "prompts" / "answer_v1.txt"

OUT_DIR_BY_CONDITION = {
    "kg2rag": REPO_ROOT / "data" / "eval" / "kg2rag",
    "temporag_kg": REPO_ROOT / "data" / "eval" / "temporag",
}

DEFAULT_QA_SOURCES = [
    REPO_ROOT / "data" / "qa" / "multihop_filtered.jsonl",
    REPO_ROOT / "data" / "qa" / "home_grown.jsonl",
]


def _load_qa(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source_file"] = p.name
                out.append(rec)
    return out


def _gold_answer(rec: dict) -> str | list[str]:
    g = rec.get("answer")
    if isinstance(g, list):
        return [str(x) for x in g]
    return str(g) if g is not None else ""


def _retrieve_all(
    qa_records: list[dict],
    index: ChunkIndex,
    kg_index: KGIndex,
    embed_client: OpenAIEmbeddingClient,
    query_cache: Cache,
    *,
    condition: str,
    k: int,
    seed_k: int,
) -> dict[str, list[dict]]:
    """Pre-compute retrieval once per question — shared across all models."""
    out: dict[str, list[dict]] = {}
    for i, rec in enumerate(qa_records, 1):
        qid = rec.get("question_id") or f"Q{i:04d}"
        qid = str(qid)
        years = rec.get("years") or []
        if condition == "kg2rag":
            res = retrieve_kg2rag(
                rec["question"], index, kg_index, embed_client, query_cache,
                k=k, seed_k=seed_k,
            )
        elif condition == "temporag_kg":
            res = retrieve_temporag_kg(
                rec["question"], years, index, kg_index, embed_client, query_cache,
                k=k, seed_k=seed_k,
            )
        else:
            raise ValueError(f"unknown condition: {condition}")
        out[qid] = res
        if i % 25 == 0 or i == len(qa_records):
            print(f"  retrieved {i}/{len(qa_records)}", flush=True)
    return out


def _run_one_model(
    config_name: str,
    qa_records: list[dict],
    retrieved: dict[str, list[dict]],
    template: str,
    cache: Cache,
    *,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    provider, model_id = MODEL_REGISTRY[config_name]
    client = build_client(config_name)

    preds: list[str] = []
    golds: list[str | list[str]] = []
    f1s: list[float] = []
    ems: list[float] = []
    n_cache = 0
    t0 = time.time()

    with preds_path.open("w", encoding="utf-8") as f_out:
        for i, rec in enumerate(qa_records, 1):
            qid = str(rec.get("question_id") or f"Q{i:04d}")
            chunks = retrieved[qid]
            try:
                res = answer_question(
                    rec["question"], chunks, client, cache,
                    template=template, model=model_id,
                )
            except Exception as exc:
                print(
                    f"  [{i}/{len(qa_records)}] {qid} FAILED: "
                    f"{type(exc).__name__}: {str(exc)[:80]}",
                    flush=True,
                )
                continue
            gold = _gold_answer(rec)
            f1, e = f1_token(res["answer"], gold), em(res["answer"], gold)
            preds.append(res["answer"])
            golds.append(gold)
            f1s.append(f1)
            ems.append(e)
            if res["cache_hit"]:
                n_cache += 1
            row = {
                "question_id": qid,
                "source": rec.get("_source_file"),
                "scope": rec.get("scope"),
                "hop_count": rec.get("hop_count"),
                "question": rec["question"],
                "gold": gold,
                "prediction": res["answer"],
                "f1": f1,
                "em": e,
                "cache_hit": res["cache_hit"],
                "parse_error": res["parse_error"],
                "retrieved_ids": [c.get("chunk_id") for c in chunks],
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(qa_records):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i:>3}/{len(qa_records)}] cache={n_cache} "
                    f"rate={rate:.1f}/s f1_rolling={sum(f1s)/len(f1s):.3f}",
                    flush=True,
                )

    summary = aggregate(preds, golds)
    summary["model"] = config_name
    summary["model_id"] = model_id
    summary["provider"] = provider
    summary["cache_hits"] = n_cache
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _write_report(summaries: list[dict], condition: str, out_root: Path) -> None:
    title = {"kg2rag": "L2 KG²RAG", "temporag_kg": "L3 TempoRAG-KG"}[condition]
    lines = [
        f"# {title} Evaluation",
        "",
        "| Model | Provider | F1 | F1 CI (95%) | EM | EM CI (95%) | n | Cache hits |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for s in summaries:
        f1_lo, f1_hi = s["f1_ci"]
        em_lo, em_hi = s["em_ci"]
        lines.append(
            f"| `{s['model']}` | {s['provider']} | "
            f"{s['f1_mean']:.3f} | [{f1_lo:.3f}, {f1_hi:.3f}] | "
            f"{s['em_mean']:.3f} | [{em_lo:.3f}, {em_hi:.3f}] | "
            f"{s['n']} | {s['cache_hits']} |"
        )
    lines += ["", f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_"]
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", required=True, choices=["kg2rag", "temporag_kg"],
        help="L2 (no temporal) or L3 (triple-level temporal filter).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Subset of configs. Valid: {sorted(MODEL_REGISTRY)}",
    )
    parser.add_argument("--qa-sources", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed-k", type=int, default=3,
                        help="Top cosine seeds to expand from (before KG walk).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()

    models = args.models or list(MODEL_REGISTRY)
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown models: {unknown}; valid: {sorted(MODEL_REGISTRY)}"
        )

    qa_paths = [Path(p) for p in (args.qa_sources or DEFAULT_QA_SOURCES)]
    qa_records = _load_qa(qa_paths)
    if args.limit is not None:
        qa_records = qa_records[: args.limit]
    print(f"Loaded {len(qa_records)} QA records from {len(qa_paths)} source(s)")

    if not (INDEX_DIR / "embeddings.npy").exists():
        raise SystemExit(f"chunk index not found at {INDEX_DIR}.")
    if not TRIPLES_PATH.exists():
        raise SystemExit(
            f"filtered triples not found at {TRIPLES_PATH}. "
            f"Run: python scripts/filter_kg_triples.py"
        )

    index = load_index(INDEX_DIR, CHUNKS_PATH)
    kg_index = load_kg_index(TRIPLES_PATH)
    print(f"KG: {len(kg_index.chunk_to_triples)} chunks w/ triples, "
          f"{len(kg_index.entity_to_chunks)} entities")

    embed_client = OpenAIEmbeddingClient()
    query_cache = Cache(QUERY_EMBED_CACHE)
    answer_cache = Cache(ANSWER_CACHE)
    template = load_answer_template(ANSWER_PROMPT)

    out_root = OUT_DIR_BY_CONDITION[args.condition]
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.condition}] retrieving top-{args.k} (seed-k={args.seed_k}) "
          f"for {len(qa_records)} questions ...")
    retrieved = _retrieve_all(
        qa_records, index, kg_index, embed_client, query_cache,
        condition=args.condition, k=args.k, seed_k=args.seed_k,
    )

    summaries: list[dict] = []
    for m in models:
        print(f"\n── [{args.condition}] Running model: {m} ──")
        summary = _run_one_model(
            m, qa_records, retrieved, template, answer_cache,
            out_dir=out_root / m,
        )
        summaries.append(summary)
        f1_lo, f1_hi = summary["f1_ci"]
        print(
            f"  → F1 {summary['f1_mean']:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]   "
            f"EM {summary['em_mean']:.3f}   n={summary['n']}"
        )

    _write_report(summaries, args.condition, out_root)
    print(f"\nReport: {(out_root / 'report.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
