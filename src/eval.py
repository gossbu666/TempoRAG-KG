"""QA evaluation: SQuAD/HotpotQA-style token F1, Exact Match, bootstrap CI, aggregate.

Normalization matches hotpot_evaluate_v1.py (lower, strip articles a/an/the,
strip punctuation, collapse whitespace). For multi-answer gold sets we take
the max score over alternatives (same convention as SQuAD).

See tasks/plan.md §5 T3.
"""

from __future__ import annotations

import os
import random
import re
import string
from collections import Counter
from typing import Iterable, Sequence


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _f1_single(pred: str, gold: str) -> float:
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)

    # SQuAD convention: if either side is empty (e.g. yes/no/impossible), treat
    # as EM — 1.0 iff both are empty after normalization, else 0.
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _as_alternatives(gold: str | Sequence[str]) -> list[str]:
    if isinstance(gold, str):
        return [gold]
    return list(gold)


def f1_token(pred: str, gold: str | Sequence[str]) -> float:
    """Token-level F1. `gold` may be a single string or a list of acceptable answers."""
    return max(_f1_single(pred, g) for g in _as_alternatives(gold))


def em(pred: str, gold: str | Sequence[str]) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    p = _normalize(pred)
    return float(any(p == _normalize(g) for g in _as_alternatives(gold)))


def bootstrap_ci(
    scores: Sequence[float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic when seeded.

    Uses RANDOM_SEED env var as default seed so results are reproducible across runs.
    """
    if not scores:
        return (0.0, 0.0)
    if seed is None:
        env = os.environ.get("RANDOM_SEED")
        seed = int(env) if env else 42
    rng = random.Random(seed)
    k = len(scores)
    means = []
    for _ in range(n):
        sample = [scores[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo_idx = int((alpha / 2) * n)
    hi_idx = int((1 - alpha / 2) * n) - 1
    hi_idx = max(hi_idx, lo_idx)
    return (means[lo_idx], means[hi_idx])


def aggregate(
    preds: Sequence[str],
    golds: Sequence[str | Sequence[str]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) length mismatch")
    f1s = [f1_token(p, g) for p, g in zip(preds, golds)]
    ems = [em(p, g) for p, g in zip(preds, golds)]
    n = len(preds)
    f1_mean = sum(f1s) / n if n else 0.0
    em_mean = sum(ems) / n if n else 0.0
    return {
        "f1_mean": f1_mean,
        "f1_ci": bootstrap_ci(f1s, n=n_bootstrap, alpha=alpha, seed=seed),
        "em_mean": em_mean,
        "em_ci": bootstrap_ci(ems, n=n_bootstrap, alpha=alpha, seed=seed),
        "n": n,
    }
