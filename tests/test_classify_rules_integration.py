"""Integration test for Stage 1 rules classifier.

Feeds a hand-crafted 10-row predictions bundle through the rule
pipeline and asserts the primary_cause each row is assigned.
"""
import json
import os
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
         "question": "Apple FY22 revenue?", "gold": "394 billion in fiscal 2022",
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
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    by_qid = {r["question_id"]: r for r in rows}
    expected = {
        "T01": "A3", "T02": "A4", "T03": "A5", "T04": "B2",
        "T05": "NF", "T06": "B4", "T07": "B5", "T08": "NF",
        "T09": None, "T10": None,
    }
    actual = {qid: by_qid[qid]["primary_cause"] for qid in expected}
    assert actual == expected, f"primary_cause mismatch:\n  expected: {expected}\n  actual:   {actual}"
