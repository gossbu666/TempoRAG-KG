"""Deterministic sampling of HotpotQA, MuSiQue questions, and 10-K chunks.

Emits:
  data/samples/hotpot_1000.json    — 500 temporal + 500 non-temporal  (v1 corpus)
  data/samples/musique_500.json    — 500 stratified across 2/3/4 hops  (v1 corpus)
  data/samples/10k_chunks.jsonl    — all chunks from 25 10-K filings   (v2 corpus)

All three outputs are byte-identical across runs (seeded Random for v1; pure
deterministic iteration order + tiktoken windowing for v2). See
tasks/plan.md §5 T1 (v1) and §6 A1 (v2).

Temporal patterns mirror the EDA classifier in src/temporal_eda.py. If the
patterns ever change, the EDA must be re-run so the counts stay consistent.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


# Must stay in sync with src/temporal_eda.py::PATTERNS
PATTERNS: dict[str, re.Pattern] = {
    "year_mention":      re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b", re.IGNORECASE),
    "when_what_year":    re.compile(r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bwhat decade\b|\bin what year\b", re.IGNORECASE),
    "before_after":      re.compile(r"\bbefore\b|\bafter\b|\bsince\b|\buntil\b|\bprior to\b|\bfollowing\b", re.IGNORECASE),
    "time_period":       re.compile(r"\bduring\b|\bin the \d{4}s\b|\bera\b|\bdecade\b|\bcentury\b|\bat the time\b", re.IGNORECASE),
    "first_last":        re.compile(r"\bfirst\b|\blast\b|\bearli(er|est)\b|\blatest\b|\bmost recent\b|\boriginal\b", re.IGNORECASE),
    "age_duration":      re.compile(r"\bhow (long|old)\b|\bage\b|\byears? (old|later|earlier|ago|apart|before|after)\b|\bduration\b", re.IGNORECASE),
    "temporal_relation": re.compile(r"\bsame (year|time|period|decade)\b|\bcontemporary\b|\bpredecessor\b|\bsuccessor\b", re.IGNORECASE),
}


def classify(question: str) -> tuple[bool, list[str]]:
    matched = [name for name, pat in PATTERNS.items() if pat.search(question)]
    return bool(matched), matched


def _largest_remainder(totals: dict[int, int], n: int) -> dict[int, int]:
    total = sum(totals.values())
    raw = {k: n * v / total for k, v in totals.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(floors.values())
    # Deterministic tie-break: fractional part desc, then key asc.
    order = sorted(totals.keys(), key=lambda k: (-(raw[k] - floors[k]), k))
    allocation = dict(floors)
    for k in order[:remaining]:
        allocation[k] += 1
    return allocation


def _write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _get_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    load_dotenv(override=False)
    env = os.environ.get("RANDOM_SEED")
    return int(env) if env else 42


def sample_hotpot(
    raw_path: Path | str,
    out_path: Path | str,
    n_temporal: int = 500,
    n_non_temporal: int = 500,
    seed: int | None = None,
) -> list[dict]:
    raw = Path(raw_path)
    out = Path(out_path)
    with raw.open("r", encoding="utf-8") as f:
        data = json.load(f)

    temporal: list[dict] = []
    non_temporal: list[dict] = []
    for item in data:
        is_t, matched = classify(item["question"])
        rec = {
            "id": item["_id"],
            "question": item["question"],
            "temporal": is_t,
            "hop_count": None,
            "patterns": matched,
        }
        (temporal if is_t else non_temporal).append(rec)

    if len(temporal) < n_temporal:
        raise ValueError(f"only {len(temporal)} temporal HotpotQA items, need {n_temporal}")
    if len(non_temporal) < n_non_temporal:
        raise ValueError(f"only {len(non_temporal)} non-temporal HotpotQA items, need {n_non_temporal}")

    # Sort each stratum by id so sampling is reproducible regardless of input order.
    temporal.sort(key=lambda r: r["id"])
    non_temporal.sort(key=lambda r: r["id"])

    rng = random.Random(_get_seed(seed))
    picked = rng.sample(temporal, n_temporal) + rng.sample(non_temporal, n_non_temporal)
    picked.sort(key=lambda r: r["id"])
    _write_json(out, picked)
    return picked


def sample_musique(
    raw_path: Path | str,
    out_path: Path | str,
    n: int = 500,
    hops: Iterable[int] = (2, 3, 4),
    seed: int | None = None,
) -> list[dict]:
    raw = Path(raw_path)
    out = Path(out_path)
    hops_list = sorted(hops)

    strata: dict[int, list[dict]] = {h: [] for h in hops_list}
    with raw.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            n_hops = len(item.get("question_decomposition", []))
            if n_hops not in strata:
                continue
            is_t, matched = classify(item["question"])
            strata[n_hops].append({
                "id": item["id"],
                "question": item["question"],
                "temporal": is_t,
                "hop_count": n_hops,
                "patterns": matched,
            })

    totals = {h: len(strata[h]) for h in hops_list}
    if sum(totals.values()) < n:
        raise ValueError(f"only {sum(totals.values())} MuSiQue items across hops {hops_list}, need {n}")
    allocation = _largest_remainder(totals, n)
    for h, k in allocation.items():
        if k > totals[h]:
            raise ValueError(f"allocation {k} exceeds available {totals[h]} for {h}-hop")

    rng = random.Random(_get_seed(seed))
    picked: list[dict] = []
    for h in hops_list:
        strata[h].sort(key=lambda r: r["id"])
        picked.extend(rng.sample(strata[h], allocation[h]))
    picked.sort(key=lambda r: r["id"])
    _write_json(out, picked)
    return picked


# -----------------------------------------------------------------------------
# 10-K chunk sampling (v2 corpus)
# -----------------------------------------------------------------------------

# Target items in a fixed order so chunk_id generation is deterministic.
# Mirrors src/parse_10k.py::TARGET_ITEMS.
_TARGET_ITEMS: tuple[str, ...] = ("1", "1A", "7", "7A", "8")

# tiktoken tokenizer used across the project for cost/length accounting.
_TOKENIZER_NAME = "cl100k_base"

# Window size and stride (overlap = 100) per docs/10k_scoping.md §3.
_CHUNK_TOKENS = 512
_CHUNK_OVERLAP = 100
_MIN_TAIL_TOKENS = 32  # skip the final window if it's this short AND fully overlaps the previous


def _get_tokenizer():
    # Deferred import so the module imports cleanly in environments where
    # tiktoken isn't installed (v1 HotpotQA/MuSiQue sampling stays usable).
    import tiktoken
    return tiktoken.get_encoding(_TOKENIZER_NAME)


def _window_token_ids(token_ids: list[int], size: int, stride: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] half-open token-id slices covering the input.

    - Fixed stride until the tail runs out.
    - The final window is kept if it extends past the previous window; dropped
      if it's fully contained AND shorter than _MIN_TAIL_TOKENS (avoids a
      duplicated 10-token leftover chunk).
    """
    if not token_ids:
        return []
    n = len(token_ids)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        windows.append((start, end))
        if end == n:
            break
        start += stride

    if len(windows) >= 2:
        prev_start, prev_end = windows[-2]
        last_start, last_end = windows[-1]
        tail_len = last_end - last_start
        if last_end <= prev_end and tail_len < _MIN_TAIL_TOKENS:
            windows.pop()
    return windows


def _chunk_section(text: str, enc) -> list[tuple[str, int]]:
    """Tokenize `text` and split into (chunk_text, token_count) tuples."""
    if not text or not text.strip():
        return []
    ids = enc.encode(text)
    out: list[tuple[str, int]] = []
    for start, end in _window_token_ids(ids, _CHUNK_TOKENS, _CHUNK_TOKENS - _CHUNK_OVERLAP):
        chunk_ids = ids[start:end]
        chunk_text = enc.decode(chunk_ids)
        out.append((chunk_text, len(chunk_ids)))
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _filing_index(manifest_path: Path) -> list[dict]:
    """Return manifest filings sorted by (ticker, fiscal_year).

    Sorting the manifest here — rather than trusting its on-disk order —
    keeps chunk ids stable even if download_10k.py is modified to emit in
    a different order.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    filings = list(manifest.get("filings", []))
    filings.sort(key=lambda r: (r["ticker"], int(r["fiscal_year"])))
    return filings


def sample_10k_chunks(
    manifest_path: Path | str = "data/10k/manifest.json",
    sections_root: Path | str = "data/10k/sections",
    out_path: Path | str | None = "data/samples/10k_chunks.jsonl",
    n: int | None = None,
) -> list[dict]:
    """Deterministically chunk every 10-K section into 512/100-token windows.

    Each record:
      {
        "chunk_id":          "<TICKER>_FY<year>_item<item>_<idx:03d>",
        "ticker":            str,
        "fy":                int,
        "item":              str,           # one of "1","1A","7","7A","8"
        "text":              str,           # decoded window text
        "sha256":            str,           # sha256(text.encode("utf-8"))
        "token_count":       int,           # actual cl100k_base token count
        "filing_date":       "YYYY-MM-DD",  # from manifest
        "period_of_report":  "YYYY-MM-DD",  # from manifest
      }

    The order of iteration is (ticker asc, fy asc, item in _TARGET_ITEMS order,
    chunk_idx asc), matching the chunk_id sort. Output is written as JSONL with
    sort_keys=True, producing byte-identical bytes on repeat runs.

    `n` truncates the stream (take the first n chunks in sort order). Pass None
    to get the whole corpus.
    """
    manifest = Path(manifest_path)
    sections_dir = Path(sections_root)
    enc = _get_tokenizer()

    records: list[dict] = []
    for filing in _filing_index(manifest):
        ticker = filing["ticker"]
        fy = int(filing["fiscal_year"])
        filing_date = filing["filing_date"]
        period = filing["period_of_report"]
        fy_dir = sections_dir / ticker / f"FY{fy}"
        for item in _TARGET_ITEMS:
            section_path = fy_dir / f"item_{item}.txt"
            if not section_path.exists():
                continue
            text = section_path.read_text(encoding="utf-8")
            for idx, (chunk_text, token_count) in enumerate(_chunk_section(text, enc)):
                records.append({
                    "chunk_id": f"{ticker}_FY{fy}_item{item}_{idx:03d}",
                    "ticker": ticker,
                    "fy": fy,
                    "item": item,
                    "text": chunk_text,
                    "sha256": _sha256_text(chunk_text),
                    "token_count": token_count,
                    "filing_date": filing_date,
                    "period_of_report": period,
                })

    records.sort(key=lambda r: r["chunk_id"])
    if n is not None:
        records = records[:n]

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return records


def main() -> None:
    # v1 HotpotQA / MuSiQue sampling — only run if inputs present so the v2
    # workflow doesn't fail on a fresh checkout without the v1 raw files.
    hp_raw = Path("data/hotpot_dev_distractor_v1.json")
    ms_raw = Path("data/musique_ans_v1.0_dev.jsonl")
    if hp_raw.exists():
        sample_hotpot(raw_path=hp_raw, out_path="data/samples/hotpot_1000.json")
    if ms_raw.exists():
        sample_musique(raw_path=ms_raw, out_path="data/samples/musique_500.json")

    if Path("data/10k/manifest.json").exists():
        sample_10k_chunks()


if __name__ == "__main__":
    main()
