"""QA evaluation: token F1, EM, LLM-Judge, BERTScore, bootstrap CI, scope aggregate.

Normalization for F1/EM matches hotpot_evaluate_v1.py (lower, strip articles
a/an/the, strip punctuation, collapse whitespace). For multi-answer gold sets
we take the max score over alternatives (same convention as SQuAD).

The LLM-Judge scores on a 0-10 integer scale matching FinReflectKG-MultiHop
(arXiv:2510.02906) for cross-paper comparability. BERTScore is used as a
secondary reference metric. `aggregate_by_scope` produces stratified
means+CIs keyed by question scope (e.g. intra-document / inter-year /
cross-company) — the temporal-lift claim depends on this breakdown.

See tasks/plan.md §5 T3 / T3.2.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from src.cache import Cache


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_IDK_RE = re.compile(
    r"^\s*(i\s+don'?t\s+know|i\s+do\s+not\s+know|no\s+information|"
    r"not\s+enough\s+information|insufficient\s+information|"
    r"cannot\s+(be\s+)?determin|unknown|n/?a)\b",
    re.I,
)


def is_idk(pred: str) -> bool:
    """Did the model decline to answer?

    Treats empty strings as IDK too — upstream API failures (reasoning models
    with exhausted max_tokens, content-filter refusals, parse errors that
    leave empty predictions) have the same downstream effect: no usable
    answer. Mixing "refused" with "failed" understates coverage by a rounding
    error, but the alternative (counting empties as valid attempts) inflates
    refusal rate with plumbing noise.
    """
    s = pred.strip() if pred else ""
    if not s:
        return True
    return bool(_IDK_RE.match(s))


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
    """Token F1 + EM + coverage.

    `f1_mean` / `em_mean` are computed over ALL rows (IDK → 0). `f1_answered`
    is computed over rows the model attempted — that's the quality-given-
    attempted number, which factors out retrieval-coverage failures from
    answer-quality failures. Both matter: coverage tells us if retrieval
    surfaced anything usable; f1_answered tells us if the model used what it
    got. Vanilla RAG on 10-K tends to suffer both.
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) length mismatch")
    f1s = [f1_token(p, g) for p, g in zip(preds, golds)]
    ems = [em(p, g) for p, g in zip(preds, golds)]
    idks = [is_idk(p) for p in preds]
    answered_f1s = [f for f, idk in zip(f1s, idks) if not idk]
    n = len(preds)
    n_answered = len(answered_f1s)
    f1_mean = sum(f1s) / n if n else 0.0
    em_mean = sum(ems) / n if n else 0.0
    cov = (n - sum(idks)) / n if n else 0.0
    f1_ans_mean = sum(answered_f1s) / n_answered if n_answered else 0.0
    return {
        "f1_mean": f1_mean,
        "f1_ci": bootstrap_ci(f1s, n=n_bootstrap, alpha=alpha, seed=seed),
        "em_mean": em_mean,
        "em_ci": bootstrap_ci(ems, n=n_bootstrap, alpha=alpha, seed=seed),
        "coverage": cov,
        "idk_rate": 1.0 - cov,
        "n_answered": n_answered,
        "f1_answered_mean": f1_ans_mean,
        "f1_answered_ci": bootstrap_ci(
            answered_f1s, n=n_bootstrap, alpha=alpha, seed=seed,
        ),
        "n": n,
    }


# -----------------------------------------------------------------------------
# LLM-Judge (0-10 integer scale, FinReflectKG-MultiHop convention)
# -----------------------------------------------------------------------------
# Kept minimal because the judge is orthogonal to F1/EM: it is a separate
# scoring function that the pilot calls per (question, gold, pred) row. Cache
# semantics mirror kg_extract: raw response in, parsed score out. That way a
# parser tweak re-scores for free.
# -----------------------------------------------------------------------------


class JudgeClient(Protocol):
    """Minimal protocol for the LLM used as judge. Matches the kg_extract
    LLMClient shape so we can reuse a single adapter (e.g. GeminiClient)."""

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str: ...


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def render_judge_prompt(template: str, question: str, gold: str, pred: str) -> str:
    """Substitute the three placeholders in `prompts/judge_v1.txt`."""
    return (
        template
        .replace("{{QUESTION}}", question)
        .replace("{{GOLD}}", gold)
        .replace("{{PRED}}", pred)
    )


def parse_judge_response(raw: str) -> tuple[int | None, str, str | None]:
    """Parse `{"score": int, "reason": str}` from a raw judge response.

    Returns (score, reason, error). Score is clamped to 0-10 after parsing.
    Tolerates a code fence or leading prose around the JSON object — we grab
    the first `{...}` we find, which matches the strict single-line contract
    in `prompts/judge_v1.txt`.
    """
    body = raw.strip()
    if body.startswith("```"):
        # Strip a ```json ... ``` fence.
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```\s*$", "", body)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(body)
        if not m:
            return None, "", f"no JSON object in response: {raw[:120]!r}"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return None, "", f"JSONDecodeError on extracted object: {e.msg}"
    if not isinstance(obj, dict):
        return None, "", f"top-level JSON must be an object, got {type(obj).__name__}"
    score = obj.get("score")
    if not isinstance(score, int) or isinstance(score, bool):
        return None, "", f"'score' must be integer, got {score!r}"
    if not 0 <= score <= 10:
        # Clamp rather than reject — Groq/Gemini occasionally emit 11/-1 under
        # rubric ambiguity; the paper's 0-10 scale is still the right summary.
        score = max(0, min(10, score))
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return score, reason, None


def score_with_judge(
    question: str,
    pred: str,
    gold: str,
    judge_client: JudgeClient,
    cache: Cache,
    *,
    prompt_template: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
) -> dict:
    """Score a single (question, pred, gold) row with the LLM-Judge.

    Returns {"score": int|None, "reason": str, "cache_hit": bool,
             "raw": str, "parse_error": str|None}.

    On parse error, score is None — the caller decides whether to retry or
    drop the row. We never raise here because a judge misformat is data,
    not a crash.
    """
    rendered = render_judge_prompt(prompt_template, question, gold, pred)
    params = {"temperature": temperature}
    key = cache.key_for(model, rendered, params)
    cached = cache.get(key)
    if cached is not None:
        raw = cached["response"]
        score, reason, err = parse_judge_response(raw)
        return {"score": score, "reason": reason, "cache_hit": True,
                "raw": raw, "parse_error": err}
    raw = judge_client.generate(rendered, temperature=temperature)
    cache.put(key, {"response": raw, "model": model})
    score, reason, err = parse_judge_response(raw)
    return {"score": score, "reason": reason, "cache_hit": False,
            "raw": raw, "parse_error": err}


# -----------------------------------------------------------------------------
# BERTScore (secondary metric)
# -----------------------------------------------------------------------------
# Deferred import so pytest can load this module without the `bert-score`
# package in the environment — matches the same pattern we use for Gemini in
# kg_extract.py. The pilot (`scripts/run_pilot.py`) is responsible for
# installing it.
# -----------------------------------------------------------------------------


def score_bertscore(
    preds: Sequence[str],
    golds: Sequence[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
) -> list[float]:
    """Return per-item BERTScore F1 (float in [0,1]).

    Uses `bert-score`'s default English model when `model_type` is None
    (roberta-large). Caller must pass single-gold strings — for multi-gold
    sets, call this once per gold alternative and take the max.
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) length mismatch")
    if not preds:
        return []
    try:
        from bert_score import score as bs_score
    except ImportError as e:
        raise ImportError(
            "bert-score is not installed. Run: pip install bert-score"
        ) from e
    kwargs: dict[str, Any] = {"lang": lang, "verbose": False}
    if model_type is not None:
        kwargs["model_type"] = model_type
    _p, _r, f1 = bs_score(list(preds), list(golds), **kwargs)
    return [float(x) for x in f1.tolist()]


# -----------------------------------------------------------------------------
# Stratified aggregation by scope
# -----------------------------------------------------------------------------


def aggregate_by_scope(
    records: Sequence[dict],
    *,
    scope_field: str = "scope",
    score_field: str = "judge_score",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Bucket `records` by `record[scope_field]` and return per-bucket stats.

    Skips records whose `score_field` is None (e.g. judge parse errors) so
    that the cell's reported `n` matches the number of scored rows, not the
    number attempted. Returns a dict keyed by scope label plus an "overall"
    entry computed over all non-None scores.

    Each entry: {"mean": float, "ci": (lo, hi), "n": int}.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for rec in records:
        score = rec.get(score_field)
        if score is None:
            continue
        scope = rec.get(scope_field, "<missing>")
        buckets[scope].append(float(score))
        all_scores.append(float(score))

    def _summary(scores: list[float]) -> dict:
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        return {
            "mean": mean,
            "ci": bootstrap_ci(scores, n=n_bootstrap, alpha=alpha, seed=seed),
            "n": n,
        }

    out = {scope: _summary(scores) for scope, scores in sorted(buckets.items())}
    out["overall"] = _summary(all_scores)
    return out
