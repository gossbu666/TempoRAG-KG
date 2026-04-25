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


def is_tersification(pred: str, gold: str, *, min_len_ratio: float = 0.2) -> bool:
    """Substring match (either direction) under a length-ratio guard.

    Length ratio = min(len_pred, len_gold) / max(...). Reject ratios
    < `min_len_ratio` so that trivially short predictions don't match
    long golds just because the short string appears inside.

    Default 0.2 is tuned against the unit tests: it accepts a 14-char
    short-form numeric answer inside a ~56-char gold sentence (ratio
    ~0.25) while still rejecting a 2-char token inside a ~43-char
    sentence (ratio ~0.05).
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
