"""Deterministic sampling of HotpotQA and MuSiQue questions.

Emits:
  data/samples/hotpot_1000.json  — 500 temporal + 500 non-temporal
  data/samples/musique_500.json  — 500 stratified across 2/3/4 hops proportional
                                   to the full dev-set hop distribution

Both files are byte-identical across runs (seeded Random + sorted inputs +
sorted outputs + json sort_keys=True). See tasks/plan.md §5 T1.

Temporal patterns mirror the EDA classifier in src/temporal_eda.py. If the
patterns ever change, the EDA must be re-run so the counts stay consistent.
"""

from __future__ import annotations

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


def main() -> None:
    sample_hotpot(
        raw_path="data/hotpot_dev_distractor_v1.json",
        out_path="data/samples/hotpot_1000.json",
    )
    sample_musique(
        raw_path="data/musique_ans_v1.0_dev.jsonl",
        out_path="data/samples/musique_500.json",
    )


if __name__ == "__main__":
    main()
