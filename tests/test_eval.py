"""Tests for src.eval — F1/EM normalization, bootstrap determinism, aggregate,
LLM-Judge parsing + caching, BERTScore deferred import, scope aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.cache import Cache
from src.eval import (
    aggregate,
    aggregate_by_scope,
    bootstrap_ci,
    em,
    f1_token,
    parse_judge_response,
    render_judge_prompt,
    score_bertscore,
    score_with_judge,
)


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


# ------------------------- render_judge_prompt -------------------------------

def test_render_judge_prompt_substitutes_all_placeholders() -> None:
    tpl = "Q: {{QUESTION}}\nG: {{GOLD}}\nP: {{PRED}}"
    out = render_judge_prompt(tpl, "who", "Obama", "Biden")
    assert "{{" not in out
    assert "Q: who" in out
    assert "G: Obama" in out
    assert "P: Biden" in out


# ------------------------- parse_judge_response -----------------------------

def test_parse_judge_valid() -> None:
    raw = '{"score": 8, "reason": "close but rounding"}'
    score, reason, err = parse_judge_response(raw)
    assert err is None
    assert score == 8
    assert reason == "close but rounding"


def test_parse_judge_strips_code_fence() -> None:
    raw = '```json\n{"score": 10, "reason": "perfect"}\n```'
    score, reason, err = parse_judge_response(raw)
    assert err is None and score == 10 and reason == "perfect"


def test_parse_judge_extracts_object_from_prose() -> None:
    # Model occasionally prepends a sentence even when told not to.
    raw = 'My verdict is below.\n{"score": 4, "reason": "wrong year"}'
    score, reason, err = parse_judge_response(raw)
    assert err is None and score == 4 and reason == "wrong year"


def test_parse_judge_clamps_out_of_range() -> None:
    raw = '{"score": 11, "reason": "overflow"}'
    score, _, err = parse_judge_response(raw)
    assert err is None and score == 10

    raw = '{"score": -2, "reason": "neg"}'
    score, _, err = parse_judge_response(raw)
    assert err is None and score == 0


def test_parse_judge_rejects_non_integer_score() -> None:
    raw = '{"score": 7.5, "reason": "half"}'
    _, _, err = parse_judge_response(raw)
    assert err is not None


def test_parse_judge_rejects_missing_score() -> None:
    raw = '{"reason": "forgot score"}'
    _, _, err = parse_judge_response(raw)
    assert err is not None


def test_parse_judge_rejects_non_json() -> None:
    raw = 'The answer is 8/10.'
    _, _, err = parse_judge_response(raw)
    assert err is not None


# ------------------------- score_with_judge (cache semantics) ---------------

class _FakeJudgeClient:
    """Records every generate() call; returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.response


def test_score_with_judge_writes_cache_on_miss(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    client = _FakeJudgeClient('{"score": 9, "reason": "ok"}')
    tpl = "Q:{{QUESTION}}\nG:{{GOLD}}\nP:{{PRED}}"
    out = score_with_judge(
        "q1", "pred", "gold", client, cache,
        prompt_template=tpl, model="fake-model",
    )
    assert out["cache_hit"] is False
    assert out["score"] == 9
    assert client.calls == 1


def test_score_with_judge_uses_cache_on_hit(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    client = _FakeJudgeClient('{"score": 9, "reason": "ok"}')
    tpl = "Q:{{QUESTION}}\nG:{{GOLD}}\nP:{{PRED}}"
    score_with_judge("q1", "p", "g", client, cache,
                     prompt_template=tpl, model="fake-model")
    client.response = '{"score": 0, "reason": "SHOULD NOT BE USED"}'
    out = score_with_judge("q1", "p", "g", client, cache,
                           prompt_template=tpl, model="fake-model")
    assert out["cache_hit"] is True
    assert out["score"] == 9
    assert client.calls == 1  # second call must be served from cache


def test_score_with_judge_returns_parse_error_without_raising(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    client = _FakeJudgeClient("not-json")
    tpl = "Q:{{QUESTION}}\nG:{{GOLD}}\nP:{{PRED}}"
    out = score_with_judge("q1", "p", "g", client, cache,
                           prompt_template=tpl, model="fake-model")
    assert out["score"] is None
    assert out["parse_error"] is not None


# ------------------------- score_bertscore ---------------------------------

def test_score_bertscore_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score_bertscore(["a", "b"], ["a"])


def test_score_bertscore_empty_returns_empty_without_import() -> None:
    # Early-return guard means we never hit the bert_score import.
    assert score_bertscore([], []) == []


# ------------------------- aggregate_by_scope ------------------------------

def test_aggregate_by_scope_buckets_correctly() -> None:
    recs = [
        {"scope": "intra", "judge_score": 10},
        {"scope": "intra", "judge_score": 8},
        {"scope": "inter-year", "judge_score": 6},
        {"scope": "inter-year", "judge_score": 4},
        {"scope": "cross-company", "judge_score": 7},
    ]
    out = aggregate_by_scope(recs, n_bootstrap=50, seed=0)
    assert set(out.keys()) == {"intra", "inter-year", "cross-company", "overall"}
    assert out["intra"]["n"] == 2
    assert out["intra"]["mean"] == pytest.approx(9.0)
    assert out["inter-year"]["mean"] == pytest.approx(5.0)
    assert out["overall"]["n"] == 5
    assert out["overall"]["mean"] == pytest.approx(7.0)


def test_aggregate_by_scope_skips_none_scores() -> None:
    recs = [
        {"scope": "intra", "judge_score": 10},
        {"scope": "intra", "judge_score": None},  # parse error row
        {"scope": "inter-year", "judge_score": 6},
    ]
    out = aggregate_by_scope(recs, n_bootstrap=50, seed=0)
    assert out["intra"]["n"] == 1
    assert out["overall"]["n"] == 2


def test_aggregate_by_scope_empty_records() -> None:
    out = aggregate_by_scope([], n_bootstrap=50, seed=0)
    assert out == {"overall": {"mean": 0.0, "ci": (0.0, 0.0), "n": 0}}
