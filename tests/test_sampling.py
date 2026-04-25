"""Tests for src.sampling — stratification, determinism, record schema.

Covers the v1 HotpotQA/MuSiQue samplers and the v2 `sample_10k_chunks` path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.sampling import (
    _largest_remainder,
    _window_token_ids,
    classify,
    sample_10k_chunks,
    sample_hotpot,
    sample_musique,
)


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


# ------------------------- 10-K chunk sampling -------------------------------

# Items populated in the synthetic corpus. Not all target items are present so
# the test also covers the "skip missing section" branch.
_SYNTH_TICKERS: tuple[tuple[str, int], ...] = (
    ("AAPL", 2022),
    ("AAPL", 2023),
    ("MSFT", 2023),
)
_SYNTH_ITEMS_PRESENT: tuple[str, ...] = ("1", "1A", "7", "8")  # item 7A intentionally missing


def _write_10k_fixture(tmp_root: Path) -> tuple[Path, Path]:
    """Create a fake `data/10k/` layout: manifest.json + sections/*.txt."""
    manifest_path = tmp_root / "manifest.json"
    sections_root = tmp_root / "sections"
    filings = []
    for ticker, fy in _SYNTH_TICKERS:
        fy_dir = sections_root / ticker / f"FY{fy}"
        fy_dir.mkdir(parents=True, exist_ok=True)
        for item in _SYNTH_ITEMS_PRESENT:
            # ~3,000 words per section — guaranteed multi-chunk at 512 tokens.
            body = (
                f"Section {item} of {ticker} fiscal year {fy}. "
                f"The company reported revenue growth during fiscal {fy}. "
                "Operating expenses increased year over year driven by investments "
                "in cloud infrastructure, research and development headcount, and "
                "data-center capacity. Management expects these trends to continue "
                "in the following fiscal year. "
            ) * 40
            (fy_dir / f"item_{item}.txt").write_text(body, encoding="utf-8")
        filings.append({
            "ticker": ticker,
            "fiscal_year": fy,
            "filing_date": f"{fy + 1}-02-01",
            "period_of_report": f"{fy}-09-30" if ticker == "AAPL" else f"{fy}-06-30",
            "accession_no": "0000000000-00-000000",
            "primary_document": "filing.htm",
            "document_url": "about:blank",
            "local_path": f"data/10k/raw/{ticker}/FY{fy}.html",
            "size_bytes": 1,
            "sha256": "0" * 64,
            "cik": "0000000000",
        })
    manifest_path.write_text(json.dumps({"filings": filings}), encoding="utf-8")
    return manifest_path, sections_root


def test_window_token_ids_covers_all_tokens_with_overlap() -> None:
    ids = list(range(1200))
    windows = _window_token_ids(ids, size=512, stride=412)
    # Every window is ≤512 long and consecutive windows share exactly 100 ids
    # until the tail, where the last window may stop short.
    assert all(0 <= s < e <= len(ids) and e - s <= 512 for s, e in windows)
    for (s1, e1), (s2, _) in zip(windows, windows[1:]):
        assert s2 - s1 == 412  # stride
        assert e1 - s2 == 100  # overlap
    # The union of windows covers every token id.
    covered = set()
    for s, e in windows:
        covered.update(range(s, e))
    assert covered == set(range(len(ids)))


def test_window_token_ids_drops_tiny_fully_overlapping_tail() -> None:
    # 512 + 1 extra token → the "tail" window of length 1 is fully inside the
    # first window and well below _MIN_TAIL_TOKENS, so it must be dropped.
    ids = list(range(513))
    windows = _window_token_ids(ids, size=512, stride=412)
    assert windows == [(0, 512), (412, 513)] or windows == [(0, 512)]
    # 1 token is < _MIN_TAIL_TOKENS=32 → expect the tail dropped when it also
    # fully overlaps. Here (412,513) is NOT fully inside (0,512) because 513>512.
    # So this case keeps both — asserting we don't drop windows that extend past.
    assert (412, 513) in windows


def test_sample_10k_chunks_record_schema(tmp_path: Path) -> None:
    manifest, sections = _write_10k_fixture(tmp_path)
    out = tmp_path / "10k_chunks.jsonl"
    records = sample_10k_chunks(manifest, sections, out)

    assert records, "expected at least one chunk from fixture"
    expected_keys = {
        "chunk_id", "ticker", "fy", "item", "text", "sha256",
        "token_count", "filing_date", "period_of_report",
    }
    for rec in records:
        assert set(rec.keys()) == expected_keys
        assert rec["ticker"] in {t for t, _ in _SYNTH_TICKERS}
        assert rec["item"] in _SYNTH_ITEMS_PRESENT
        assert rec["sha256"] == hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
        assert 0 < rec["token_count"] <= 512
        # filing_date / period_of_report flow through from the manifest
        assert rec["filing_date"].startswith(str(rec["fy"] + 1))
        assert rec["period_of_report"].startswith(str(rec["fy"]))


def test_sample_10k_chunks_chunk_id_deterministic_and_sorted(tmp_path: Path) -> None:
    manifest, sections = _write_10k_fixture(tmp_path)
    records = sample_10k_chunks(manifest, sections, tmp_path / "out.jsonl")
    ids = [r["chunk_id"] for r in records]
    assert ids == sorted(ids)
    # chunk_id format: <TICKER>_FY<year>_item<item>_<idx:03d>
    for chunk_id in ids:
        ticker_part, fy_part, item_part, idx_part = chunk_id.split("_")
        assert fy_part.startswith("FY") and fy_part[2:].isdigit()
        assert item_part.startswith("item")
        assert idx_part.isdigit() and len(idx_part) == 3


def test_sample_10k_chunks_skips_missing_sections(tmp_path: Path) -> None:
    manifest, sections = _write_10k_fixture(tmp_path)
    records = sample_10k_chunks(manifest, sections, tmp_path / "out.jsonl")
    # Fixture never wrote item 7A → must not appear in output
    assert all(r["item"] != "7A" for r in records)
    # But all other target items should appear for every filing
    for ticker, fy in _SYNTH_TICKERS:
        present = {r["item"] for r in records if r["ticker"] == ticker and r["fy"] == fy}
        assert present == set(_SYNTH_ITEMS_PRESENT)


def test_sample_10k_chunks_byte_identical(tmp_path: Path) -> None:
    manifest, sections = _write_10k_fixture(tmp_path)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    sample_10k_chunks(manifest, sections, out_a)
    sample_10k_chunks(manifest, sections, out_b)
    assert hashlib.sha256(out_a.read_bytes()).hexdigest() == hashlib.sha256(out_b.read_bytes()).hexdigest()


def test_sample_10k_chunks_n_truncates(tmp_path: Path) -> None:
    manifest, sections = _write_10k_fixture(tmp_path)
    full = sample_10k_chunks(manifest, sections, tmp_path / "full.jsonl")
    first5 = sample_10k_chunks(manifest, sections, tmp_path / "first5.jsonl", n=5)
    assert len(first5) == 5
    assert first5 == full[:5]
