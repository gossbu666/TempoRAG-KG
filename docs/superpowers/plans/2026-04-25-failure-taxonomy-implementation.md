# Failure Taxonomy Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4-stage reusable pipeline that labels every `(model, condition, question)` prediction with a failure category, produces count tables + worked examples, and reports Cohen's κ against a 30-case human sample.

**Architecture:** Three core stages (rules → LLM → aggregate) + an optional fourth (κ sample) that each read/write JSONL files on disk so any stage can be rerun without recomputing the others. Rules stage is deterministic and free; LLM stage only touches the ~40-60 % of rows the rules couldn't resolve; aggregate emits CSV tables + markdown examples + report.

**Tech Stack:** Python 3.9 (existing venv), `openai>=1.0` (already in repo), pytest (existing `tests/`), `src.cache.Cache` (existing), `src.eval.f1_token` (existing). No new dependencies.

**Design source:** [`docs/superpowers/specs/2026-04-25-failure-taxonomy-design.md`](../specs/2026-04-25-failure-taxonomy-design.md).

## File Structure

| Path | Role | Owner task |
|---|---|---|
| `src/taxonomy.py` | Shared helpers: normalizers, IDK regex, n-gram extractor, rule precedence, category codes. Keeps rule logic out of script top-level so unit tests can target it. | T1 |
| `tests/test_taxonomy.py` | Unit tests for every helper in `src/taxonomy.py`. | T1 |
| `scripts/classify_failures_rules.py` | Stage 1. Loads all predictions from `data/eval/*/predictions.jsonl`, joins QA records, applies deterministic rules, writes `rules_stage.jsonl`. | T2 |
| `tests/test_classify_rules_integration.py` | Integration test: hand-crafted 10-row predictions file → expected labels per rule. | T2 |
| `prompts/classify_failure_v1.txt` | LLM prompt template for Stage 2. | T3 |
| `scripts/classify_failures_llm.py` | Stage 2. Reads `rules_stage.jsonl`, calls `gpt-4o-mini` on rows with null `primary_cause`, writes `classified_predictions.jsonl`. | T3 |
| `scripts/classify_failures_agg.py` | Stage 3. Reads `classified_predictions.jsonl`, emits 4 CSVs + `examples.md` + `report.md`. | T4 |
| `scripts/kappa_sample.py` | Stage 4. Stratified sample → interactive CLI → computes Cohen's κ → updates `report.md`. | T5 |

**Inputs already on disk:**

- `data/eval/{vanilla,timefilter,kg2rag,temporag}/<model>/predictions.jsonl` (L3 sweep in progress — plan tolerates it landing mid-implementation).
- `data/samples/10k_chunks.jsonl` — 7,467 chunks.
- `data/qa/{home_grown,multihop_filtered}.jsonl` — 129 labeled QA records; plus `synth_multihop_v1.jsonl` after sub-(d) vet.

**Outputs land under:** `data/eval/failure_taxonomy/`.

---

## Task 1: `src/taxonomy.py` — shared helpers + tests

**Files:**
- Create: `src/taxonomy.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_taxonomy.py`:

```python
"""Unit tests for src/taxonomy helpers."""
from src.taxonomy import (
    CATEGORY_CODES, RULE_PRECEDENCE,
    normalize, is_idk, gold_ngrams,
    any_ngram_in_chunks, is_tersification,
)


def test_normalize_lowercases_and_strips_punct():
    assert normalize("  Hello, World!  ") == "hello world"
    assert normalize("$5,234.00") == "5 234 00"
    assert normalize("I don't know.") == "i don t know"


def test_is_idk_matches_common_forms():
    assert is_idk("I don't know")
    assert is_idk("i dont know")
    assert is_idk("I don’t know.")
    assert is_idk("I don't know the answer yet")
    assert not is_idk("The company does not disclose this")
    assert not is_idk("")


def test_gold_ngrams_returns_3plus_word_ngrams():
    ngrams = gold_ngrams("Apple reported revenue of $394 billion in fiscal 2022")
    assert "apple reported revenue" in ngrams
    assert "394 billion in" in ngrams
    # stop-word-only ngrams should be rejected
    assert "of the in" not in ngrams


def test_gold_ngrams_rejects_short_gold():
    assert gold_ngrams("Yes") == set()
    assert gold_ngrams("$5B") == set()


def test_any_ngram_in_chunks_substring_match():
    ngrams = {"apple reported revenue"}
    chunks_with = ["In fiscal 2022, Apple reported revenue of $394B."]
    chunks_without = ["Microsoft posted $198B in fiscal 2022."]
    assert any_ngram_in_chunks(ngrams, chunks_with)
    assert not any_ngram_in_chunks(ngrams, chunks_without)


def test_is_tersification_substring_both_directions():
    # Pred is inside gold; length ratio ok.
    assert is_tersification(
        pred="$53,803 million",
        gold="Cisco's total revenue for fiscal 2024 was $53,803 million.",
    )
    # Gold inside pred.
    assert is_tersification(
        pred="The answer, based on the 10-K, is $53,803 million.",
        gold="$53,803 million",
    )
    # Totally different strings.
    assert not is_tersification(pred="$50,000 million", gold="$53,803 million")
    # Substring but length ratio too small.
    assert not is_tersification(pred="10", gold="The revenue grew by 10 percent year-on-year.")


def test_category_codes_and_precedence_consistent():
    assert set(RULE_PRECEDENCE) <= set(CATEGORY_CODES)
    # A5 beats NF beats A3 beats A4 beats B4 beats B5 beats B2 beats B3
    order = RULE_PRECEDENCE
    assert order.index("A5") < order.index("NF")
    assert order.index("NF") < order.index("A3")
    assert order.index("A4") < order.index("B4")
```

- [ ] **Step 2: Run tests to verify they fail**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_taxonomy.py -v
```
Expected: every test fails with `ImportError: cannot import name ...` or equivalent.

- [ ] **Step 3: Implement `src/taxonomy.py`**

```python
"""Shared helpers for failure-taxonomy classification.

Keep the rule logic in one small module so Stage-1 classification script
stays thin (it just orchestrates I/O and precedence), and so every rule
is individually unit-testable.
"""
from __future__ import annotations

import re
import string

# Category codes used throughout the pipeline.
CATEGORY_CODES: list[str] = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "NF",
]

# Precedence of deterministic rules at Stage 1. First match wins. A5
# (parse error) beats everything — garbage output can't be NF. NF must
# beat A3 so high-F1 predictions are never mislabeled as tersification.
RULE_PRECEDENCE: list[str] = [
    "A5", "NF", "A3", "A4", "B4", "B5", "B2", "B3",
]

_IDK_RE = re.compile(r"(?i)i\s*don[\'’]?t\s*know")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Stop-word set — intentionally small. Only discarded when the *entire*
# ngram is stop-words; a stop-word alongside content words is fine.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with",
})


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return " ".join(s.split())


def is_idk(s: str) -> bool:
    """True if the prediction matches the 'I don't know' family."""
    if not s:
        return False
    return _IDK_RE.search(s) is not None


def gold_ngrams(gold: str, n: int = 3) -> set[str]:
    """Return all `n`-word n-grams from gold, normalized, excluding
    ngrams that are entirely stop-words. Returns empty set if gold has
    fewer than n tokens."""
    norm = normalize(gold)
    tokens = norm.split()
    if len(tokens) < n:
        return set()
    out: set[str] = set()
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        if all(w in _STOPWORDS for w in window):
            continue
        out.add(" ".join(window))
    return out


def any_ngram_in_chunks(ngrams: set[str], chunk_texts: list[str]) -> bool:
    """True if any ngram appears (case-folded substring) in any chunk text."""
    if not ngrams:
        return False
    normed_chunks = [normalize(t) for t in chunk_texts]
    for ng in ngrams:
        for ct in normed_chunks:
            if ng in ct:
                return True
    return False


def is_tersification(pred: str, gold: str, *, min_len_ratio: float = 0.3) -> bool:
    """Substring match (either direction) under a length-ratio guard.

    Length ratio = min(len_pred, len_gold) / max(...). Reject ratios
    < `min_len_ratio` so that trivially short predictions don't match
    long golds just because the short string appears inside.
    """
    if not pred or not gold:
        return False
    a = normalize(pred)
    b = normalize(gold)
    if not a or not b:
        return False
    if a not in b and b not in a:
        return False
    lo = min(len(a), len(b))
    hi = max(len(a), len(b))
    return (lo / hi) >= min_len_ratio
```

- [ ] **Step 4: Run tests to verify they pass**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_taxonomy.py -v
```
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```
git add src/taxonomy.py tests/test_taxonomy.py
git commit -m "feat(taxonomy): shared helpers for Stage 1 failure-classifier rules"
```

---

## Task 2: `scripts/classify_failures_rules.py` — Stage 1 + integration test

**Files:**
- Create: `scripts/classify_failures_rules.py`
- Create: `tests/test_classify_rules_integration.py`
- Modify: none

- [ ] **Step 1: Write the failing integration test**

`tests/test_classify_rules_integration.py`:

```python
"""Integration test for Stage 1 rules classifier.

Feeds a hand-crafted 10-row predictions bundle through the rule
pipeline and asserts the primary_cause each row is assigned.
"""
import json
import subprocess
from pathlib import Path


def _write_predictions(dir_: Path, rows: list[dict]) -> None:
    (dir_ / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_stage1_rules_label_each_category(tmp_path: Path):
    # Minimal chunk store used by A4/NF rules.
    chunks = [
        {"chunk_id": "AAPL_FY2022_item7_001", "ticker": "AAPL", "fy": 2022,
         "item": "7", "text": "Apple reported revenue of 394 billion in fiscal 2022."},
        {"chunk_id": "MSFT_FY2022_item7_002", "ticker": "MSFT", "fy": 2022,
         "item": "7", "text": "Microsoft reported total revenue of 198 billion."},
    ]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )

    # 10 fake predictions covering every rule-matched category (A3, A4, A5,
    # B2, B3, B4, B5, NF) and two ambiguous rows (left null for Stage 2).
    preds = [
        {"question_id": "T01", "scope": "intra", "hop_count": 1,
         "tickers": ["AAPL"], "years": [2022],
         "question": "Apple FY22 revenue?", "gold": "Apple reported revenue of 394 billion in fiscal 2022",
         "prediction": "394 billion", "f1": 0.22, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001"]},  # A3 tersification
        {"question_id": "T02", "scope": "intra", "hop_count": 1,
         "tickers": ["AAPL"], "years": [2022],
         "question": "Apple FY22 revenue?", "gold": "394 billion",
         "prediction": "I don't know", "f1": 0.0, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001"]},  # A4 IDK when answerable
        {"question_id": "T03", "scope": "intra", "hop_count": 1,
         "tickers": ["AAPL"], "years": [2022],
         "question": "?", "gold": "394 billion",
         "prediction": "raw text no JSON", "f1": 0.0,
         "parse_error": "JSONDecodeError: line 1", "retrieved_ids": []},  # A5 parse error
        {"question_id": "T04", "scope": "forward_looking", "hop_count": 2,
         "tickers": ["META"], "years": [2023],
         "question": "What will happen?", "gold": "unclear",
         "prediction": "unknown", "f1": 0.0, "parse_error": None,
         "retrieved_ids": []},  # B2 forward_looking
        {"question_id": "T05", "scope": "fiscal_vs_calendar", "hop_count": 1,
         "tickers": ["MSFT"], "years": [2023],
         "question": "Revenue for FY ending June 2023?", "gold": "211 billion",
         "prediction": "211 billion", "f1": 0.7, "parse_error": None,
         "retrieved_ids": []},  # B3 fiscal_vs_calendar precedence - but NF wins (f1>=0.5)
        {"question_id": "T06", "scope": "intra", "hop_count": 1,
         "tickers": ["NFLX"], "years": [2022],  # NFLX not in corpus
         "question": "?", "gold": "?",
         "prediction": "?", "f1": 0.0, "parse_error": None,
         "retrieved_ids": []},  # B4 out-of-scope
        {"question_id": "T07", "scope": "cross_company", "hop_count": 3,
         "tickers": ["AAPL", "MSFT"], "years": [2022],
         "question": "Compare Apple and Microsoft FY22 revenue",
         "gold": "Apple 394, Microsoft 198",
         "prediction": "Apple 394 vs MSFT 198", "f1": 0.25, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001", "MSFT_FY2022_item7_002"]},  # B5 cross-filing
        {"question_id": "T08", "scope": "intra", "hop_count": 1,
         "tickers": ["AAPL"], "years": [2022],
         "question": "?", "gold": "Apple reported revenue of 394 billion in fiscal 2022",
         "prediction": "Apple reported revenue of 394 billion in fiscal 2022.",
         "f1": 0.98, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001"]},  # NF high F1
        {"question_id": "T09", "scope": "intra", "hop_count": 1,
         "tickers": ["AAPL"], "years": [2022],
         "question": "?", "gold": "obscure unretrieved fact",
         "prediction": "Apple reported something else",
         "f1": 0.1, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001"]},  # Ambiguous → null (Stage 2)
        {"question_id": "T10", "scope": "intra", "hop_count": 2,
         "tickers": ["AAPL"], "years": [2022],
         "question": "?", "gold": "obscure fact",
         "prediction": "Apple made stuff up", "f1": 0.05, "parse_error": None,
         "retrieved_ids": ["AAPL_FY2022_item7_001"]},  # Ambiguous → null
    ]
    cond_dir = tmp_path / "L0" / "model-x"
    cond_dir.mkdir(parents=True)
    _write_predictions(cond_dir, preds)

    # Run the stage.
    out = tmp_path / "out.jsonl"
    cp = subprocess.run(
        ["venv/bin/python", "scripts/classify_failures_rules.py",
         "--predictions-root", str(tmp_path),
         "--chunks", str(tmp_path / "chunks.jsonl"),
         "--out", str(out)],
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    by_qid = {r["question_id"]: r for r in rows}
    assert by_qid["T01"]["primary_cause"] == "A3"
    assert by_qid["T02"]["primary_cause"] == "A4"
    assert by_qid["T03"]["primary_cause"] == "A5"
    assert by_qid["T04"]["primary_cause"] == "B2"
    assert by_qid["T05"]["primary_cause"] == "NF"  # high F1 short-circuits B3
    assert by_qid["T06"]["primary_cause"] == "B4"
    assert by_qid["T07"]["primary_cause"] == "B5"
    assert by_qid["T08"]["primary_cause"] == "NF"
    assert by_qid["T09"]["primary_cause"] is None  # Stage 2 will decide
    assert by_qid["T10"]["primary_cause"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_classify_rules_integration.py -v
```
Expected: `FileNotFoundError: scripts/classify_failures_rules.py` or `returncode != 0`.

- [ ] **Step 3: Implement `scripts/classify_failures_rules.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_classify_rules_integration.py -v
```
Expected: test passes.

- [ ] **Step 5: Commit**

```
git add scripts/classify_failures_rules.py tests/test_classify_rules_integration.py
git commit -m "feat(taxonomy): Stage 1 rules classifier + 10-case integration test"
```

---

## Task 3: LLM prompt + `scripts/classify_failures_llm.py`

**Files:**
- Create: `prompts/classify_failure_v1.txt`
- Create: `scripts/classify_failures_llm.py`

- [ ] **Step 1: Write `prompts/classify_failure_v1.txt`**

```
You are classifying why a 10-K QA system failed on one question.

INPUT:
  Question   : {QUESTION}
  Gold answer: {GOLD}
  Model pred : {PREDICTION}
  Top-k retrieved chunks (each is a passage from a 10-K filing):
{CONTEXT}
  F1 score   : {F1}

CATEGORIES (pick exactly one for `primary`):
  A1 retrieval_miss  - The gold fact is NOT present in any of the
    retrieved chunks above, so the model never saw it.
  A2 hallucination   - The gold fact IS present in at least one chunk,
    but the prediction asserts something else (wrong number, wrong
    entity, invented fact).
  B1 corpus_limit    - The gold fact is NOT something a 10-K filing
    would normally contain, regardless of retrieval. Examples:
    stock price on a specific date, analyst opinion, board-meeting
    content, future-dated guidance not yet disclosed.

OPTIONAL `secondary` ∈ {A1, A2, A3, A4, B1, B2, B3, B4, B5} or null —
include only if a distinct second failure mode clearly also applies
(e.g. the model both missed retrieval AND the corpus would not have
answered it anyway).

Respond with ONLY JSON on one line, no fences, no prose:
{"primary": "A1" | "A2" | "B1",
 "secondary": "<code>" | null,
 "reason": "<=25 words>"}
```

- [ ] **Step 2: Write the failing test**

`tests/test_classify_llm.py`:

```python
"""Test Stage-2 LLM parsing and prompt rendering on a stub client."""
import json
from pathlib import Path
from scripts.classify_failures_llm import render_prompt, parse_response


def test_render_prompt_fills_all_placeholders():
    template = Path("prompts/classify_failure_v1.txt").read_text()
    row = {
        "question": "What is Apple's FY22 revenue?",
        "gold": "$394B",
        "prediction": "$500B",
        "f1": 0.0,
        "retrieved_ids": ["AAPL_FY2022_item7_001"],
    }
    chunks = {"AAPL_FY2022_item7_001":
              {"text": "Apple reported revenue of $394B in fiscal 2022.",
               "ticker": "AAPL", "fy": 2022, "item": "7"}}
    out = render_prompt(template, row, chunks)
    assert "{QUESTION}" not in out
    assert "{GOLD}" not in out
    assert "{PREDICTION}" not in out
    assert "{CONTEXT}" not in out
    assert "{F1}" not in out
    assert "$394B" in out
    assert "AAPL_FY2022_item7_001" in out


def test_parse_response_accepts_clean_json():
    primary, secondary, reason = parse_response(
        '{"primary": "A2", "secondary": null, "reason": "Model invented a number."}'
    )
    assert primary == "A2"
    assert secondary is None
    assert "invented" in reason.lower()


def test_parse_response_strips_fences():
    raw = '```json\n{"primary":"A1","secondary":null,"reason":"miss"}\n```'
    primary, _, _ = parse_response(raw)
    assert primary == "A1"


def test_parse_response_rejects_unknown_code():
    primary, _, _ = parse_response('{"primary":"XX","secondary":null,"reason":"nope"}')
    assert primary is None  # unknown code should fall through
```

- [ ] **Step 3: Implement `scripts/classify_failures_llm.py`**

```python
"""Stage 2 of the failure-taxonomy classifier: LLM for ambiguous rows.

Reads `rules_stage.jsonl`. Every row with `primary_cause == null` is
sent to `gpt-4o-mini` (temperature 0) with the prompt at
`prompts/classify_failure_v1.txt`. Responses are cached on
`(model, rendered_prompt, temperature=0)` via `src.cache.Cache` so
re-runs are free. Writes the merged `classified_predictions.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache
from src.taxonomy import CATEGORY_CODES

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TEMP = 0.0
DEFAULT_MAX_TOKENS = 200
CACHE_DIR = Path("data/cache/failure_classify")

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_VALID_PRIMARY = {"A1", "A2", "B1"}


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def render_prompt(template: str, row: dict, chunks: dict[str, dict]) -> str:
    """Fill `{QUESTION}`, `{GOLD}`, `{PREDICTION}`, `{CONTEXT}`, `{F1}`."""
    ctx_blocks: list[str] = []
    for cid in row.get("retrieved_ids") or []:
        c = chunks.get(cid)
        if not c:
            continue
        header = f"    [{cid} | {c.get('ticker','?')} FY{c.get('fy','?')} Item {c.get('item','?')}]"
        ctx_blocks.append(header + "\n    " + c.get("text", "").replace("\n", " ")[:900])
    context = "\n".join(ctx_blocks) if ctx_blocks else "    (no chunks retrieved)"
    gold = row.get("gold")
    if isinstance(gold, list):
        gold = " | ".join(str(g) for g in gold)
    f1 = row.get("f1")
    return (template
            .replace("{QUESTION}", str(row.get("question", "")))
            .replace("{GOLD}", str(gold or ""))
            .replace("{PREDICTION}", str(row.get("prediction", "")))
            .replace("{CONTEXT}", context)
            .replace("{F1}", f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1)))


def parse_response(raw: str) -> tuple[str | None, str | None, str]:
    """Return (primary, secondary, reason) or (None, None, raw) on failure."""
    m = _FENCE_RE.match(raw.strip())
    text = m.group(1).strip() if m else raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, None, raw
    if not isinstance(obj, dict):
        return None, None, raw
    primary = obj.get("primary")
    secondary = obj.get("secondary")
    reason = str(obj.get("reason", "")).strip()
    if primary not in _VALID_PRIMARY:
        return None, None, reason or raw
    if secondary is not None and secondary not in CATEGORY_CODES:
        secondary = None
    return primary, secondary, reason


def _call_with_cache(prompt: str, client, cache: Cache, model: str,
                     temperature: float, max_tokens: int) -> str:
    params = {"temperature": temperature, "max_tokens": max_tokens}
    key = cache.key_for(model, prompt, params)
    cached = cache.get(key)
    if cached is not None:
        return cached["response"]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    cache.put(key, {"response": raw, "model": model})
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-jsonl",
                        default="data/eval/failure_taxonomy/rules_stage.jsonl")
    parser.add_argument("--chunks",
                        default="data/samples/10k_chunks.jsonl")
    parser.add_argument("--prompt",
                        default="prompts/classify_failure_v1.txt")
    parser.add_argument("--out",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    template = Path(args.prompt).read_text(encoding="utf-8")
    chunks = _load_chunks(Path(args.chunks))
    cache = Cache(CACHE_DIR)

    rows: list[dict] = []
    with open(args.rules_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ambiguous = [r for r in rows if r.get("primary_cause") is None]
    if args.limit is not None:
        ambiguous = ambiguous[: args.limit]
    print(f"Total rows: {len(rows)}  ambiguous (Stage 2): {len(ambiguous)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    resolved: dict[tuple[str, str, str], dict] = {}
    t0 = time.time()
    for i, row in enumerate(ambiguous, 1):
        prompt = render_prompt(template, row, chunks)
        try:
            raw = _call_with_cache(prompt, client, cache,
                                   args.model, DEFAULT_TEMP, DEFAULT_MAX_TOKENS)
        except Exception as exc:
            print(f"  [{i}/{len(ambiguous)}] {row['question_id']} FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            continue
        primary, secondary, reason = parse_response(raw)
        key = (str(row["question_id"]), row["condition"], row["model"])
        resolved[key] = {"primary_cause": primary,
                         "secondary_cause": secondary,
                         "reason": reason}
        if i % 50 == 0 or i == len(ambiguous):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(ambiguous)}] rate={rate:.1f}/s",
                  flush=True)

    # Merge resolutions back into the full row set and write out.
    with open(args.out, "w", encoding="utf-8") as fout:
        for row in rows:
            key = (str(row["question_id"]), row["condition"], row["model"])
            if key in resolved:
                row = {**row, **resolved[key]}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_classify_llm.py -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```
git add prompts/classify_failure_v1.txt scripts/classify_failures_llm.py tests/test_classify_llm.py
git commit -m "feat(taxonomy): Stage 2 LLM classifier (gpt-4o-mini) + prompt + parser tests"
```

---

## Task 4: `scripts/classify_failures_agg.py` — count matrices + examples + report

**Files:**
- Create: `scripts/classify_failures_agg.py`
- Create: `tests/test_classify_agg.py`

- [ ] **Step 1: Write the failing test**

`tests/test_classify_agg.py`:

```python
"""Test Stage 3 aggregation — count matrices and example selection."""
import csv
import json
import subprocess
from pathlib import Path


def _write_classified(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")


def test_aggregate_counts_and_examples(tmp_path: Path):
    rows = [
        {"question_id": "Q1", "model": "m1", "condition": "L0",
         "primary_cause": "A1", "secondary_cause": None, "f1": 0.1,
         "scope": "intra", "hop_count": 1, "question": "Q1?",
         "gold": "a", "prediction": "b", "retrieved_ids": []},
        {"question_id": "Q2", "model": "m1", "condition": "L0",
         "primary_cause": "A2", "secondary_cause": None, "f1": 0.3,
         "scope": "intra", "hop_count": 1, "question": "Q2?",
         "gold": "a", "prediction": "c", "retrieved_ids": []},
        {"question_id": "Q3", "model": "m2", "condition": "L1",
         "primary_cause": "NF", "secondary_cause": None, "f1": 0.9,
         "scope": "cross_company", "hop_count": 3,
         "question": "Q3?", "gold": "a", "prediction": "a",
         "retrieved_ids": []},
    ]
    inp = tmp_path / "classified.jsonl"
    _write_classified(inp, rows)

    out_dir = tmp_path / "agg"
    cp = subprocess.run(
        ["venv/bin/python", "scripts/classify_failures_agg.py",
         "--classified", str(inp), "--out-dir", str(out_dir)],
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr

    # by_model.csv should have 2 rows (m1, m2).
    with (out_dir / "by_model.csv").open() as f:
        reader = list(csv.DictReader(f))
    models = {r["model"] for r in reader}
    assert models == {"m1", "m2"}
    m1 = next(r for r in reader if r["model"] == "m1")
    assert int(m1["A1"]) == 1
    assert int(m1["A2"]) == 1

    # examples.md should have a section per category present in data.
    examples = (out_dir / "examples.md").read_text()
    assert "## A1" in examples
    assert "## A2" in examples
    assert "## NF" in examples

    # report.md should exist.
    assert (out_dir / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_classify_agg.py -v
```
Expected: FAIL (script doesn't exist).

- [ ] **Step 3: Implement `scripts/classify_failures_agg.py`**

```python
"""Stage 3: aggregate classified predictions into tables, examples, report.

Reads `classified_predictions.jsonl`. Emits:
  - by_model.csv     : rows=models    cols=CATEGORY_CODES + total
  - by_condition.csv : rows=conditions
  - by_scope.csv     : rows=scope labels
  - by_hop.csv       : rows=hop counts (1..4)
  - examples.md      : one section per category, with the prediction
                      whose F1 is closest to the median within that bucket.
  - report.md        : headline counts + links to the above. κ section
                      placeholder is filled later by `kappa_sample.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.taxonomy import CATEGORY_CODES

AXES = ["by_model", "by_condition", "by_scope", "by_hop"]
AXIS_FIELD = {"by_model": "model", "by_condition": "condition",
              "by_scope": "scope", "by_hop": "hop_count"}


def _load(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _axis_matrix(rows: list[dict], field: str) -> dict[str, Counter]:
    """{axis_value: Counter over category codes}."""
    grid: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        key = r.get(field)
        if key is None:
            continue
        key = str(key)
        cause = r.get("primary_cause") or "unlabeled"
        grid[key][cause] += 1
    return grid


def _write_axis_csv(grid: dict[str, Counter], field: str, out: Path) -> None:
    cats = CATEGORY_CODES + ["unlabeled"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([field] + cats + ["total"])
        for key in sorted(grid):
            counts = grid[key]
            total = sum(counts.values())
            w.writerow([key] + [counts.get(c, 0) for c in cats] + [total])


def _pick_example(rows: list[dict], cause: str) -> dict | None:
    """Return the row whose f1 is closest to the median of its bucket.

    Returns None if the bucket is empty.
    """
    bucket = [r for r in rows if r.get("primary_cause") == cause]
    if not bucket:
        return None
    f1s = [float(r.get("f1") or 0.0) for r in bucket]
    med = statistics.median(f1s)
    return min(bucket, key=lambda r: abs(float(r.get("f1") or 0.0) - med))


def _examples_md(rows: list[dict]) -> str:
    lines: list[str] = ["# Failure Examples", "",
                        "One representative prediction per category "
                        "(closest to the median F1 within that bucket).",
                        ""]
    for c in CATEGORY_CODES:
        ex = _pick_example(rows, c)
        if ex is None:
            continue
        lines.append(f"## {c}")
        lines.append("")
        lines.append(f"- **Question:** {ex.get('question')}")
        lines.append(f"- **Gold:** {ex.get('gold')}")
        lines.append(f"- **Prediction:** `{ex.get('prediction')}`")
        lines.append(f"- **F1:** {ex.get('f1')}")
        lines.append(f"- **Scope / hop:** {ex.get('scope')} / {ex.get('hop_count')}")
        lines.append(f"- **Condition × model:** {ex.get('condition')} / {ex.get('model')}")
        reason = ex.get("reason")
        if reason:
            lines.append(f"- **Classifier rationale:** {reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(rows: list[dict]) -> str:
    n = len(rows)
    counts = Counter(r.get("primary_cause") or "unlabeled" for r in rows)
    nf = counts.get("NF", 0)
    ambiguous_left = counts.get("unlabeled", 0)
    lines = [
        "# Failure Taxonomy — Aggregate Report",
        "",
        f"Total classified predictions: **{n}**",
        f"Non-failures (NF): **{nf}** ({100*nf/max(n,1):.1f}%)",
        f"Unlabeled remaining: **{ambiguous_left}** (expected 0 after Stage 2)",
        "",
        "## Headline counts by category",
        "",
        "| Code | Count | % |",
        "|------|------:|---:|",
    ]
    for c in CATEGORY_CODES + ["unlabeled"]:
        n_c = counts.get(c, 0)
        lines.append(f"| {c} | {n_c} | {100*n_c/max(n,1):.1f}% |")
    lines += [
        "",
        "## Tables",
        "",
        "- [by_model.csv](by_model.csv)",
        "- [by_condition.csv](by_condition.csv)",
        "- [by_scope.csv](by_scope.csv)",
        "- [by_hop.csv](by_hop.csv)",
        "- [examples.md](examples.md)",
        "",
        "## Reliability",
        "",
        "(Filled in by `scripts/kappa_sample.py`.)",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--out-dir",
                        default="data/eval/failure_taxonomy")
    args = parser.parse_args()

    rows = _load(Path(args.classified))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ax in AXES:
        grid = _axis_matrix(rows, AXIS_FIELD[ax])
        _write_axis_csv(grid, AXIS_FIELD[ax], out_dir / f"{ax}.csv")

    (out_dir / "examples.md").write_text(_examples_md(rows), encoding="utf-8")
    (out_dir / "report.md").write_text(_report_md(rows), encoding="utf-8")
    print(f"Wrote {len(AXES)} CSVs + examples.md + report.md to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_classify_agg.py -v
```
Expected: test passes.

- [ ] **Step 5: Commit**

```
git add scripts/classify_failures_agg.py tests/test_classify_agg.py
git commit -m "feat(taxonomy): Stage 3 aggregation — count matrices + examples + report"
```

---

## Task 5: `scripts/kappa_sample.py` — interactive reliability CLI

**Files:**
- Create: `scripts/kappa_sample.py`
- Create: `tests/test_kappa.py`

- [ ] **Step 1: Write the failing test (κ computation only)**

`tests/test_kappa.py`:

```python
"""Test the Cohen's κ helper. Interactive loop is covered manually."""
from scripts.kappa_sample import cohen_kappa


def test_kappa_perfect_agreement():
    k = cohen_kappa(["A", "B", "A", "C"], ["A", "B", "A", "C"])
    assert abs(k - 1.0) < 1e-9


def test_kappa_zero_on_independent():
    # 50/50 splits with random pairing: κ around 0.
    k = cohen_kappa(["A", "B"] * 50, ["A", "B"] * 25 + ["B", "A"] * 25)
    assert -0.1 < k < 0.1


def test_kappa_worse_than_chance_is_negative():
    k = cohen_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"])
    assert k < 0
```

- [ ] **Step 2: Run test to verify it fails**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_kappa.py -v
```
Expected: `ImportError: cannot import name 'cohen_kappa'`.

- [ ] **Step 3: Implement `scripts/kappa_sample.py`**

```python
"""Stage 4 (optional): Cohen's κ reliability check on a stratified sample.

Picks 6 predictions from each of the 5 most-populated Stage-2-ambiguous
categories (target 30 rows). Walks the user through a CLI review loop
and computes Cohen's κ against the LLM labels. Appends the result to
`report.md`.

Categories offered to the user: A1, A2, B1, and the two most frequent
rule-labeled categories in the current run (dynamically resolved).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.taxonomy import CATEGORY_CODES

DEFAULT_N_PER_CATEGORY = 6


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Inline Cohen's κ — no sklearn dependency.

    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement rate, p_e = expected agreement by chance.
    """
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    expected = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _interpret(kappa: float) -> str:
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def _sample(rows: list[dict], n_per: int, seed: int) -> list[dict]:
    """Pick up to `n_per` rows from each of 5 target categories."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("primary_cause")
        if c is None:
            continue
        by_cat.setdefault(c, []).append(r)

    # Target categories: the 3 Stage-2 ones + top 2 rule-labeled.
    rule_counts = Counter(
        r.get("primary_cause") for r in rows
        if r.get("primary_cause") not in {"A1", "A2", "B1", None}
    )
    top_rule = [c for c, _ in rule_counts.most_common(2)]
    targets = ["A1", "A2", "B1"] + top_rule

    rng = random.Random(seed)
    out: list[dict] = []
    for c in targets:
        pool = by_cat.get(c, [])
        if not pool:
            continue
        rng.shuffle(pool)
        out.extend(pool[:n_per])
    return out


def _display(row: dict) -> None:
    print("═" * 70)
    print(f"QID {row['question_id']}   condition={row['condition']} model={row['model']}")
    print(f"scope={row.get('scope')}  hop={row.get('hop_count')}  f1={row.get('f1')}")
    print(f"Q: {row.get('question')}")
    print(f"A (gold): {row.get('gold')}")
    print(f"Pred: {row.get('prediction')}")
    reason = row.get("reason")
    if reason:
        print(f"LLM reason: {reason}")
    print(f"LLM label: {row['primary_cause']}  (secondary: {row.get('secondary_cause')})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--out",
                        default="data/eval/failure_taxonomy/kappa_sample.jsonl")
    parser.add_argument("--report",
                        default="data/eval/failure_taxonomy/report.md")
    parser.add_argument("--n-per-category", type=int, default=DEFAULT_N_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.classified) if l.strip()]
    sample = _sample(rows, args.n_per_category, args.seed)
    if not sample:
        raise SystemExit("no rows available for sampling (empty classified file?)")

    labeled: list[dict] = []
    print(f"You will review {len(sample)} predictions. "
          f"For each, type one of: {CATEGORY_CODES} or 'skip'.\n")
    try:
        for i, row in enumerate(sample, 1):
            print(f"\n[{i}/{len(sample)}]")
            _display(row)
            while True:
                choice = input(f"Your label ({'|'.join(CATEGORY_CODES)} or skip): ").strip().upper()
                if choice == "SKIP":
                    break
                if choice in CATEGORY_CODES:
                    labeled.append({**row, "human_label": choice})
                    break
                print("  ? unknown code; try again.")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — saving what we have.")

    Path(args.out).write_text(
        "\n".join(json.dumps(r) for r in labeled) + "\n", encoding="utf-8"
    )
    if not labeled:
        raise SystemExit("no labels collected; κ not computed.")

    llm = [r["primary_cause"] for r in labeled]
    human = [r["human_label"] for r in labeled]
    k = cohen_kappa(llm, human)
    interp = _interpret(k)
    print(f"\nCohen's κ over {len(labeled)} rows: {k:.3f} ({interp})")

    # Rewrite the reliability section of report.md.
    report_path = Path(args.report)
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        block = (
            "## Reliability\n\n"
            f"Cohen's κ (LLM vs. human) on {len(labeled)} stratified samples: "
            f"**{k:.3f}** ({interp}).\n\n"
            f"Sample file: [kappa_sample.jsonl]({Path(args.out).name}).\n"
        )
        # Replace the placeholder if it's there; else append.
        if "## Reliability" in text:
            start = text.index("## Reliability")
            end = text.find("\n## ", start + 1)
            end = end if end != -1 else len(text)
            text = text[:start] + block + ("\n" + text[end:] if end != len(text) else "\n")
        else:
            text = text.rstrip() + "\n\n" + block
        report_path.write_text(text, encoding="utf-8")
        print(f"Report updated: {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_kappa.py -v
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```
git add scripts/kappa_sample.py tests/test_kappa.py
git commit -m "feat(taxonomy): Stage 4 κ reliability CLI + inline cohen_kappa"
```

---

## Task 6: End-to-end smoke test — real L0 predictions only

**Files touched:** none (operational only).

- [ ] **Step 1: Run Stage 1 on L0 only**

```
PYTHONPATH=. venv/bin/python scripts/classify_failures_rules.py \
    --predictions-root data/eval \
    --out /tmp/rules_stage_l0only.jsonl
```
Expected: prints `Total predictions: N` equal to `7 × 129 × <n_conditions_present>`; per-category counts printed; file exists.

- [ ] **Step 2: Run Stage 2 on a 10-row slice (cost-controlled smoke)**

```
PYTHONPATH=. venv/bin/python scripts/classify_failures_llm.py \
    --rules-jsonl /tmp/rules_stage_l0only.jsonl \
    --out /tmp/classified_l0only.jsonl \
    --limit 10
```
Expected: `Total rows: N  ambiguous (Stage 2): M`, then loops through at most 10. Output file contains all rows (not just 10) — Stage 2 only re-labels the subset.

- [ ] **Step 3: Run Stage 3**

```
PYTHONPATH=. venv/bin/python scripts/classify_failures_agg.py \
    --classified /tmp/classified_l0only.jsonl \
    --out-dir /tmp/agg_l0only
```
Expected: writes 4 CSVs + examples.md + report.md.

- [ ] **Step 4: Eyeball the output**

```
head /tmp/agg_l0only/by_model.csv
head -50 /tmp/agg_l0only/report.md
```
Sanity checks:
- `by_model.csv` has one row per model.
- Sum of counts per model ≈ 129 × number of conditions in the sweep root.
- `report.md` headline counts add up to total predictions.

- [ ] **Step 5: Run full test suite**

```
PYTHONPATH=. venv/bin/python -m pytest tests/test_taxonomy.py \
    tests/test_classify_rules_integration.py \
    tests/test_classify_llm.py \
    tests/test_classify_agg.py \
    tests/test_kappa.py -v
```
Expected: all passing.

- [ ] **Step 6: Commit if anything changed**

```
git status
# no-op commit not needed; if unit tests added fixtures, commit them.
```

---

## Checkpoint: classifier ready for full run

- [ ] All unit tests pass.
- [ ] Smoke pipeline runs end-to-end on L0 predictions.
- [ ] No ambiguous rows remain in `classified_predictions.jsonl` after full Stage 2 run (or a clear accounting of why any do).
- [ ] Review: does `report.md` narrative match the category counts?

## Task 7: Full run on complete eval suite (L0+L1+L2+L3)

(Run this task *after* L3 sweep finishes.)

- [ ] **Step 1: Run Stage 1 across all conditions**

```
PYTHONPATH=. venv/bin/python scripts/classify_failures_rules.py
```
Expected: processes 7 × 4 × 129 = 3,612 rows (or more if sub-(d) Qs are already integrated).

- [ ] **Step 2: Run Stage 2 to completion (no --limit)**

```
PYTHONPATH=. nohup venv/bin/python scripts/classify_failures_llm.py \
    > data/eval/failure_taxonomy/llm_run.log 2>&1 &
```
Expected: completes in ~20-40 minutes depending on ambiguous count; gpt-4o-mini cost ≤ $0.50.

- [ ] **Step 3: Run Stage 3 aggregate**

```
PYTHONPATH=. venv/bin/python scripts/classify_failures_agg.py
```

- [ ] **Step 4: Run Stage 4 κ sample (interactive)**

```
PYTHONPATH=. venv/bin/python scripts/kappa_sample.py
```
Interactive review of 30 rows. κ printed at end and written to `report.md`.

- [ ] **Step 5: Commit outputs**

```
git add data/eval/failure_taxonomy/
git commit -m "data(taxonomy): first full classification run (7 models × 4 conditions)"
```

---

## Verification

End-to-end acceptance:

- [ ] `data/eval/failure_taxonomy/classified_predictions.jsonl` has 1 row per input prediction, every row has a `primary_cause ∈ CATEGORY_CODES`.
- [ ] Four CSVs exist and their totals sum to `len(classified_predictions.jsonl)`.
- [ ] `examples.md` has a section per non-empty category.
- [ ] `report.md` Reliability section shows a κ value ≥ 0.4 (if < 0.4, iterate on Stage 2 prompt).
- [ ] Plan document matches spec: categories (10 + NF), precedence, threshold 0.5 for tersification, both per-prediction labels (primary + optional secondary).
