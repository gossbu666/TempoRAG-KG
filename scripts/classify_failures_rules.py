"""Stage 1 of the failure-taxonomy classifier: deterministic rules.

Walks every predictions.jsonl under `--predictions-root/<condition>/<model>/`
(conditions = L0/L1/L2/L3 per the plan), applies rules in the precedence
order defined in `src.taxonomy.RULE_PRECEDENCE`, and writes a flat
`rules_stage.jsonl` at `--out`. Rows the rules can't resolve get
`primary_cause=None` for Stage 2 to decide.

Predictions files across conditions have slightly different fields (some
lack hop_count / years). We resolve those from the QA record files
(`--qa` can repeat) so every output row is fully populated.

Corpus tickers and years are hard-coded (10 tickers × 2019-2024) to match
the 10-K chunk set on disk.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.taxonomy import (
    RULE_PRECEDENCE,
    any_ngram_in_chunks,
    gold_ngrams,
    is_idk,
    is_tersification,
)

CORPUS_TICKERS = {"AAPL", "ADBE", "AMZN", "CSCO", "GOOGL",
                  "INTC", "META", "MSFT", "NVDA", "ORCL"}
CORPUS_YEARS = {2019, 2020, 2021, 2022, 2023, 2024}

CONDITION_DIR_MAP = {
    "vanilla": "L0",
    "timefilter": "L1",
    "kg2rag": "L2",
    "temporag": "L3",
}


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def _load_qa(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[str(r.get("question_id"))] = r
    return out


def _gather_predictions(root: Path) -> list[tuple[str, str, Path]]:
    """Return (condition_label, model, predictions_file) triples."""
    found: list[tuple[str, str, Path]] = []
    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir():
            continue
        label = CONDITION_DIR_MAP.get(cond_dir.name, cond_dir.name)
        for model_dir in sorted(cond_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            preds = model_dir / "predictions.jsonl"
            if preds.exists():
                found.append((label, model_dir.name, preds))
    return found


def _classify_row(
    row: dict,
    *,
    chunks: dict[str, dict],
    qa: dict[str, dict],
) -> tuple[str | None, str | None]:
    """Return (primary_cause, rule_matched) per the precedence list. If no
    rule fires, primary_cause is None and Stage 2 handles it."""
    f1 = float(row.get("f1") or 0.0)
    pred = str(row.get("prediction") or "")
    gold_raw = row.get("gold")
    gold = gold_raw if isinstance(gold_raw, str) else " ".join(gold_raw or [])
    parse_error = row.get("parse_error")
    scope = row.get("scope") or ""
    retrieved_ids = row.get("retrieved_ids") or []
    chunk_texts = [chunks[cid]["text"] for cid in retrieved_ids if cid in chunks]

    # QA-record fallback for fields missing on old prediction files.
    qa_rec = qa.get(str(row.get("question_id"))) or {}
    hop_count = row.get("hop_count") or qa_rec.get("hop_count") or 0
    tickers = row.get("tickers") or qa_rec.get("tickers") or []
    years = row.get("years") or qa_rec.get("years") or []

    gold_grams = gold_ngrams(gold)
    idk = is_idk(pred)

    for code in RULE_PRECEDENCE:
        if code == "A5":
            if parse_error:
                return "A5", "A5"
        elif code == "NF":
            if f1 >= 0.5:
                return "NF", "NF"
            if idk and not any_ngram_in_chunks(gold_grams, chunk_texts):
                return "NF", "NF"
        elif code == "A3":
            if not idk and is_tersification(pred, gold) and f1 < 0.5:
                return "A3", "A3"
        elif code == "A4":
            if idk and any_ngram_in_chunks(gold_grams, chunk_texts):
                return "A4", "A4"
        elif code == "B4":
            if tickers and not set(tickers) <= CORPUS_TICKERS:
                return "B4", "B4"
            if years and not set(years) <= CORPUS_YEARS:
                return "B4", "B4"
        elif code == "B5":
            if hop_count >= 3 and scope in {"cross_company", "inter_year"}:
                return "B5", "B5"
        elif code == "B2":
            if scope == "forward_looking":
                return "B2", "B2"
        elif code == "B3":
            if scope == "fiscal_vs_calendar":
                return "B3", "B3"
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root",
                        default="data/eval",
                        help="Directory containing condition subdirs.")
    parser.add_argument("--chunks",
                        default="data/samples/10k_chunks.jsonl")
    parser.add_argument("--qa", nargs="*",
                        default=["data/qa/home_grown.jsonl",
                                 "data/qa/multihop_filtered.jsonl",
                                 "data/qa/synth_multihop_v1.jsonl"])
    parser.add_argument("--out",
                        default="data/eval/failure_taxonomy/rules_stage.jsonl")
    args = parser.parse_args()

    root = Path(args.predictions_root)
    chunks = _load_chunks(Path(args.chunks))
    qa = _load_qa([Path(p) for p in args.qa])

    triples = _gather_predictions(root)
    if not triples:
        raise SystemExit(f"no predictions found under {root}")
    print(f"Scanning {len(triples)} (condition,model) pairs under {root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_labeled = 0
    by_cat: dict[str, int] = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as fout:
        for cond, model, p in triples:
            with p.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    n_total += 1
                    cause, rule = _classify_row(row, chunks=chunks, qa=qa)
                    if cause is not None:
                        n_labeled += 1
                        by_cat[cause] += 1
                    record = {
                        "question_id": row.get("question_id"),
                        "condition": cond,
                        "model": model,
                        "primary_cause": cause,
                        "secondary_cause": None,
                        "rule_matched": rule,
                        "f1": row.get("f1"),
                        "scope": row.get("scope"),
                        "hop_count": row.get("hop_count") or qa.get(
                            str(row.get("question_id")), {}).get("hop_count"),
                        "question": row.get("question"),
                        "gold": row.get("gold"),
                        "prediction": row.get("prediction"),
                        "retrieved_ids": row.get("retrieved_ids") or [],
                        "parse_error": row.get("parse_error"),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total predictions: {n_total}")
    print(f"Rule-labeled: {n_labeled} ({100*n_labeled/max(n_total,1):.1f}%)")
    print(f"Ambiguous (Stage-2): {n_total - n_labeled}")
    for c in sorted(by_cat):
        print(f"  {c}: {by_cat[c]}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
