"""Tests for src.eval — F1/EM normalization, bootstrap determinism, aggregate."""

from __future__ import annotations

import math

import pytest

from src.eval import aggregate, bootstrap_ci, em, f1_token


# ------------------------- F1: known-answer spot checks -----------------------

def test_f1_identical_tokens() -> None:
    assert f1_token("Paris", "Paris") == pytest.approx(1.0)


def test_f1_articles_stripped() -> None:
    # After normalization: "quick brown fox" == "quick brown fox"
    assert f1_token("The quick brown fox", "a quick brown fox") == pytest.approx(1.0)


def test_f1_punctuation_stripped() -> None:
    assert f1_token("New York, USA!", "New York USA") == pytest.approx(1.0)


def test_f1_partial_overlap() -> None:
    # pred tokens: {new, york}, gold tokens: {new, york, city}
    # P = 2/2 = 1.0, R = 2/3, F1 = 2*1*(2/3)/(1 + 2/3) = 0.8
    assert f1_token("New York", "New York City") == pytest.approx(0.8, abs=1e-3)


def test_f1_no_overlap() -> None:
    assert f1_token("cat", "dog") == 0.0


def test_f1_duplicate_tokens_use_multiset() -> None:
    # pred: [the, the, cat] -> normalized [cat] (articles stripped)
    # gold: [cat]
    # F1 = 1.0
    assert f1_token("the the cat", "cat") == pytest.approx(1.0)


# ------------------------- F1: multi-answer gold -----------------------------

def test_f1_multi_gold_takes_max() -> None:
    preds = "Barack Obama"
    golds = ["Joe Biden", "Barack Obama", "Donald Trump"]
    assert f1_token(preds, golds) == pytest.approx(1.0)


def test_f1_multi_gold_best_partial() -> None:
    # best alt is "New York City" -> F1 = 0.8, other is 0
    assert f1_token("New York", ["Paris", "New York City"]) == pytest.approx(0.8, abs=1e-3)


# ------------------------- EM -------------------------------------------------

def test_em_after_normalization() -> None:
    assert em("The Paris.", "paris") == 1.0


def test_em_false_on_different() -> None:
    assert em("London", "Paris") == 0.0


def test_em_multi_gold() -> None:
    assert em("Obama", ["Trump", "Obama"]) == 1.0


# ------------------------- Empty / edge cases --------------------------------

def test_f1_both_empty_after_normalization() -> None:
    # "the" normalizes to empty; both empty → treat as EM match = 1.0
    assert f1_token("the", "the") == 1.0


def test_f1_one_empty_after_normalization() -> None:
    assert f1_token("the", "Paris") == 0.0


# ------------------------- bootstrap_ci --------------------------------------

def test_bootstrap_ci_constant_scores() -> None:
    lo, hi = bootstrap_ci([0.5] * 20, n=200, seed=0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_deterministic_with_seed() -> None:
    scores = [0.0, 0.3, 0.5, 0.7, 1.0, 0.2, 0.8, 0.6, 0.4, 0.9]
    ci_a = bootstrap_ci(scores, n=500, seed=123)
    ci_b = bootstrap_ci(scores, n=500, seed=123)
    assert ci_a == ci_b


def test_bootstrap_ci_contains_mean() -> None:
    scores = [0.0, 0.3, 0.5, 0.7, 1.0, 0.2, 0.8, 0.6, 0.4, 0.9]
    mean = sum(scores) / len(scores)
    lo, hi = bootstrap_ci(scores, n=1000, seed=42)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


# ------------------------- aggregate -----------------------------------------

def test_aggregate_returns_required_keys() -> None:
    out = aggregate(["Paris"], ["Paris"], n_bootstrap=100, seed=0)
    for key in ("f1_mean", "f1_ci", "em_mean", "em_ci", "n"):
        assert key in out
    assert out["n"] == 1
    assert out["f1_mean"] == pytest.approx(1.0)
    assert out["em_mean"] == pytest.approx(1.0)
    assert isinstance(out["f1_ci"], tuple) and len(out["f1_ci"]) == 2


def test_aggregate_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        aggregate(["a"], ["a", "b"])


def test_aggregate_mean_f1_matches_hand_calc() -> None:
    preds = ["Paris", "New York", "cat"]
    golds = ["Paris", "New York City", "dog"]  # F1 = 1.0, 0.8, 0.0
    out = aggregate(preds, golds, n_bootstrap=100, seed=0)
    expected_mean = (1.0 + 0.8 + 0.0) / 3
    assert out["f1_mean"] == pytest.approx(expected_mean, abs=1e-3)
    assert out["em_mean"] == pytest.approx(1 / 3, abs=1e-6)
