"""Test Stage 3 aggregation — count matrices and example selection."""
import csv
import json
import os
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
        env={**os.environ, "PYTHONPATH": "."},
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
