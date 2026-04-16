"""Tests for src.sampling — stratification, determinism, record schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.sampling import _largest_remainder, classify, sample_hotpot, sample_musique


# ------------------------- classifier -----------------------------------------

def test_classify_detects_year() -> None:
    is_t, patterns = classify("In 1999 the team won.")
    assert is_t
    assert "year_mention" in patterns


def test_classify_detects_when() -> None:
    is_t, patterns = classify("When did the war end?")
    assert is_t
    assert "when_what_year" in patterns


def test_classify_non_temporal() -> None:
    is_t, patterns = classify("What color is the sky?")
    assert not is_t
    assert patterns == []


def test_classify_matches_multiple_patterns() -> None:
    is_t, patterns = classify("What year before 1990 was the first event?")
    assert is_t
    assert set(patterns) >= {"year_mention", "before_after", "first_last"}


# ------------------------- largest remainder allocation ----------------------

def test_largest_remainder_sums_to_n() -> None:
    totals = {2: 1252, 3: 760, 4: 405}
    alloc = _largest_remainder(totals, 500)
    assert sum(alloc.values()) == 500


def test_largest_remainder_proportional() -> None:
    # 2:3:5 ratio -> 20:30:50 of 100
    totals = {2: 20, 3: 30, 4: 50}
    alloc = _largest_remainder(totals, 100)
    assert alloc == {2: 20, 3: 30, 4: 50}


def test_largest_remainder_ties_are_deterministic() -> None:
    totals = {2: 100, 3: 100, 4: 100}
    alloc = _largest_remainder(totals, 7)
    # Sum must equal 7; with all tied fractions, lower keys get the extras.
    assert sum(alloc.values()) == 7
    # Each key gets floor(7/3)=2, then 1 extra goes to lowest key.
    assert alloc[2] >= alloc[3] >= alloc[4]


# ------------------------- HotpotQA fixture sampling -------------------------

def _write_hotpot_fixture(path: Path, n_temporal: int, n_non_temporal: int) -> None:
    items = []
    for i in range(n_temporal):
        items.append({
            "_id": f"t{i:04d}",
            "question": f"In {1990 + (i % 30)} what event occurred at site {i}?",
            "answer": "x",
            "type": "bridge",
            "context": [],
            "supporting_facts": [],
            "level": "medium",
        })
    for i in range(n_non_temporal):
        items.append({
            "_id": f"n{i:04d}",
            "question": f"What is item number {i}?",
            "answer": "x",
            "type": "bridge",
            "context": [],
            "supporting_facts": [],
            "level": "medium",
        })
    path.write_text(json.dumps(items), encoding="utf-8")


def test_sample_hotpot_counts(tmp_path: Path) -> None:
    raw = tmp_path / "hp.json"
    out = tmp_path / "sample.json"
    _write_hotpot_fixture(raw, n_temporal=50, n_non_temporal=50)
    picked = sample_hotpot(raw, out, n_temporal=20, n_non_temporal=20, seed=42)
    assert len(picked) == 40
    assert sum(1 for r in picked if r["temporal"]) == 20
    assert sum(1 for r in picked if not r["temporal"]) == 20


def test_sample_hotpot_record_schema(tmp_path: Path) -> None:
    raw = tmp_path / "hp.json"
    out = tmp_path / "sample.json"
    _write_hotpot_fixture(raw, n_temporal=20, n_non_temporal=20)
    picked = sample_hotpot(raw, out, n_temporal=5, n_non_temporal=5, seed=42)
    for rec in picked:
        assert set(rec.keys()) == {"id", "question", "temporal", "hop_count", "patterns"}
        assert isinstance(rec["id"], str)
        assert isinstance(rec["question"], str)
        assert isinstance(rec["temporal"], bool)
        assert rec["hop_count"] is None
        assert isinstance(rec["patterns"], list)


def test_sample_hotpot_byte_identical(tmp_path: Path) -> None:
    raw = tmp_path / "hp.json"
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    _write_hotpot_fixture(raw, n_temporal=60, n_non_temporal=60)
    sample_hotpot(raw, out_a, n_temporal=20, n_non_temporal=20, seed=42)
    sample_hotpot(raw, out_b, n_temporal=20, n_non_temporal=20, seed=42)
    h_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert h_a == h_b


def test_sample_hotpot_different_seed_differs(tmp_path: Path) -> None:
    raw = tmp_path / "hp.json"
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    _write_hotpot_fixture(raw, n_temporal=60, n_non_temporal=60)
    sample_hotpot(raw, out_a, n_temporal=20, n_non_temporal=20, seed=1)
    sample_hotpot(raw, out_b, n_temporal=20, n_non_temporal=20, seed=2)
    assert out_a.read_bytes() != out_b.read_bytes()


def test_sample_hotpot_too_few_raises(tmp_path: Path) -> None:
    raw = tmp_path / "hp.json"
    out = tmp_path / "sample.json"
    _write_hotpot_fixture(raw, n_temporal=5, n_non_temporal=5)
    with pytest.raises(ValueError):
        sample_hotpot(raw, out, n_temporal=10, n_non_temporal=10, seed=42)


# ------------------------- MuSiQue fixture sampling --------------------------

def _write_musique_fixture(path: Path, counts: dict[int, int]) -> None:
    lines = []
    for h, n in counts.items():
        for i in range(n):
            q = f"When did event {h}-{i} happen in 1999?" if i % 2 == 0 else f"What is thing {h}-{i}?"
            item = {
                "id": f"{h}hop__{i:05d}",
                "question": q,
                "answer": "x",
                "answerable": True,
                "paragraphs": [],
                "question_decomposition": [{"id": j, "question": "", "answer": "", "paragraph_support_idx": 0} for j in range(h)],
                "answer_aliases": [],
            }
            lines.append(json.dumps(item))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_sample_musique_stratification(tmp_path: Path) -> None:
    raw = tmp_path / "ms.jsonl"
    out = tmp_path / "sample.json"
    _write_musique_fixture(raw, {2: 1252, 3: 760, 4: 405})
    picked = sample_musique(raw, out, n=500, seed=42)
    assert len(picked) == 500
    by_hop: dict[int, int] = {}
    for rec in picked:
        by_hop[rec["hop_count"]] = by_hop.get(rec["hop_count"], 0) + 1
    # Expected from largest_remainder on totals 1252:760:405 allocating 500
    # 2: 500*1252/2417 = 258.99 -> 259
    # 3: 500*760/2417  = 157.22 -> 157
    # 4: 500*405/2417  = 83.78  -> 84
    assert by_hop == {2: 259, 3: 157, 4: 84}


def test_sample_musique_record_schema(tmp_path: Path) -> None:
    raw = tmp_path / "ms.jsonl"
    out = tmp_path / "sample.json"
    _write_musique_fixture(raw, {2: 100, 3: 60, 4: 40})
    picked = sample_musique(raw, out, n=50, seed=42)
    for rec in picked:
        assert set(rec.keys()) == {"id", "question", "temporal", "hop_count", "patterns"}
        assert rec["hop_count"] in (2, 3, 4)
        assert isinstance(rec["temporal"], bool)


def test_sample_musique_byte_identical(tmp_path: Path) -> None:
    raw = tmp_path / "ms.jsonl"
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    _write_musique_fixture(raw, {2: 300, 3: 200, 4: 100})
    sample_musique(raw, out_a, n=100, seed=42)
    sample_musique(raw, out_b, n=100, seed=42)
    assert hashlib.sha256(out_a.read_bytes()).hexdigest() == hashlib.sha256(out_b.read_bytes()).hexdigest()


def test_sample_musique_output_sorted_by_id(tmp_path: Path) -> None:
    raw = tmp_path / "ms.jsonl"
    out = tmp_path / "sample.json"
    _write_musique_fixture(raw, {2: 100, 3: 60, 4: 40})
    picked = sample_musique(raw, out, n=40, seed=42)
    ids = [r["id"] for r in picked]
    assert ids == sorted(ids)
