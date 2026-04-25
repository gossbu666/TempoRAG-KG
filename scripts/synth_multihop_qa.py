"""T6: Seed-based multi-hop QA synthesis for sub-(d) dataset expansion.

Strategy
--------
For each (scope, hop_count, target_count) cell:
  1. Sample {hop_count} chunks from the 10-K corpus that satisfy the
     scope's cross-chunk constraint (e.g. different tickers for
     cross_company, different fiscal years for inter_year).
  2. Feed the chunks to gpt-4o with the synthesis prompt
     (`prompts/synth_multihop_v1.txt`); the prompt enforces the hop
     contract ("remove any chunk → unanswerable").
  3. Parse the JSON response, validate hop_count/scope/tickers match,
     assign a stable question_id, write to `data/qa/synth_pool_v1.jsonl`.

All LLM calls are cached on (model, rendered prompt, temperature). A
re-run with the same seed bundle is free.

Target mix (over-generate 130 → vet → accept 85):

| scope              | hop=3 | hop=4 |
|--------------------|-------|-------|
| cross_company      |   45  |   15  |
| inter_year         |   30  |    8  |
| intra              |   18  |    —  |
| fiscal_vs_calendar |    8  |    —  |
| forward_looking    |    4  |    —  |
|                    |  105  |   25  |

Tickers cover all 10 corpus companies; per-ticker balance is applied as a
soft goal via seeded sampling (seed=42).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
PROMPT_PATH = REPO_ROOT / "prompts" / "synth_multihop_v1.txt"
OUT_PATH = REPO_ROOT / "data" / "qa" / "synth_pool_v1.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "synth"

DEFAULT_MODEL = "gpt-4o-2024-11-20"
SEED = 42

# Tickers whose fiscal year ends in a non-December month — used for the
# fiscal_vs_calendar scope bucket. AAPL (Sep), MSFT (Jun), CSCO (late Jul),
# ORCL (May), ADBE (early Dec but labeled as fiscal year distinct), NVDA (late Jan).
NON_DEC_FY = {"AAPL", "MSFT", "CSCO", "ORCL", "NVDA", "ADBE"}

# Target mix per (scope, hop_count) — sum = 130.
TARGETS: dict[tuple[str, int], int] = {
    ("cross_company", 3): 45,
    ("inter_year", 3): 30,
    ("intra", 3): 18,
    ("fiscal_vs_calendar", 3): 8,
    ("forward_looking", 3): 4,
    ("cross_company", 4): 15,
    ("inter_year", 4): 8,
}


def _load_chunks(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _sample_bundle(
    chunks: list[dict], scope: str, hop: int, rng: random.Random
) -> list[dict] | None:
    """Return a list of `hop` chunks matching the scope's contract, or None
    if the corpus can't satisfy the constraint."""
    if scope == "cross_company":
        # hop distinct tickers, all in the same year (year is ambient).
        by_ticker_year: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for c in chunks:
            by_ticker_year[(c["ticker"], c["fy"])].append(c)
        years = sorted({fy for (_, fy) in by_ticker_year})
        rng.shuffle(years)
        for y in years:
            tickers = [tk for (tk, fy) in by_ticker_year if fy == y]
            if len(tickers) < hop:
                continue
            picks = rng.sample(tickers, hop)
            return [rng.choice(by_ticker_year[(tk, y)]) for tk in picks]
        return None

    if scope == "inter_year":
        by_ticker: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for c in chunks:
            by_ticker[c["ticker"]][c["fy"]].append(c)
        tickers = [tk for tk, yrs in by_ticker.items() if len(yrs) >= hop]
        rng.shuffle(tickers)
        for tk in tickers:
            years = list(by_ticker[tk])
            if len(years) < hop:
                continue
            ys = rng.sample(years, hop)
            return [rng.choice(by_ticker[tk][y]) for y in ys]
        return None

    if scope == "intra":
        by_ticker_year: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for c in chunks:
            by_ticker_year[(c["ticker"], c["fy"])].append(c)
        candidates = [(k, v) for k, v in by_ticker_year.items()
                      if len({c["item"] for c in v}) >= hop]
        rng.shuffle(candidates)
        for (tk, y), pool in candidates:
            items_seen: set[str] = set()
            picks: list[dict] = []
            rng.shuffle(pool)
            for c in pool:
                if c["item"] in items_seen:
                    continue
                picks.append(c)
                items_seen.add(c["item"])
                if len(picks) == hop:
                    return picks
        return None

    if scope == "fiscal_vs_calendar":
        # Same ticker (non-Dec), one fiscal year; hop chunks across items.
        # Intentionally similar to intra but restricted to fiscal-ambiguous tickers.
        by_ticker_year: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for c in chunks:
            if c["ticker"] not in NON_DEC_FY:
                continue
            by_ticker_year[(c["ticker"], c["fy"])].append(c)
        candidates = [(k, v) for k, v in by_ticker_year.items()
                      if len({c["item"] for c in v}) >= hop]
        rng.shuffle(candidates)
        for (tk, y), pool in candidates:
            items_seen: set[str] = set()
            picks: list[dict] = []
            rng.shuffle(pool)
            for c in pool:
                if c["item"] in items_seen:
                    continue
                picks.append(c)
                items_seen.add(c["item"])
                if len(picks) == hop:
                    return picks
        return None

    if scope == "forward_looking":
        # At least one item 1A or item 7 chunk (forward-looking prose).
        fl_items = {"1A", "7"}
        fl = [c for c in chunks if c["item"] in fl_items]
        if len(fl) < hop:
            return None
        return rng.sample(fl, hop)

    raise ValueError(f"unknown scope: {scope}")


def _render_prompt(template: str, bundle: list[dict], scope: str, hop: int) -> str:
    tickers = sorted({c["ticker"] for c in bundle})
    years = sorted({c["fy"] for c in bundle})
    chunk_blocks = []
    for i, c in enumerate(bundle, 1):
        chunk_blocks.append(
            f"### Chunk {i} — id={c['chunk_id']}\n"
            f"Ticker: {c['ticker']}  FY: {c['fy']}  Item: {c['item']}\n\n"
            f"{c['text']}\n"
        )
    rendered = template
    rendered = rendered.replace("{N_CHUNKS}", str(hop))
    rendered = rendered.replace("{SCOPE}", scope)
    rendered = rendered.replace("{TICKERS}", str(tickers))
    rendered = rendered.replace("{YEARS}", str(years))
    rendered = rendered.replace("{TICKERS_JSON}", json.dumps(tickers))
    rendered = rendered.replace("{YEARS_JSON}", json.dumps(years))
    for i, c in enumerate(bundle, 1):
        rendered = rendered.replace(f"{{CHUNK_{i}_ID}}", c["chunk_id"])
    rendered = rendered.replace("{CHUNK_BLOCKS}", "\n".join(chunk_blocks))
    return rendered


def _synth_one(prompt: str, model: str, client, cache: Cache) -> dict | None:
    """Call the LLM, cache by prompt; return the parsed JSON or None on failure."""
    params = {"model": model, "temperature": 0.4}
    key = cache.key_for(model, prompt, params)
    cached = cache.get(key)
    if cached is not None:
        raw = cached["raw"]
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content
        cache.put(key, {"raw": raw, "model": model})
    # Strip ```json ... ``` fences if present.
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl != -1 else text
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate(rec: dict, expected_scope: str, expected_hop: int,
              bundle: list[dict]) -> str | None:
    if rec.get("question") is None:
        return rec.get("reason") or "null_question"
    q = rec.get("question", "")
    a = rec.get("answer", "")
    if not q or not a:
        return "empty_q_or_a"
    if rec.get("scope") != expected_scope:
        return f"scope_mismatch:{rec.get('scope')}"
    if int(rec.get("hop_count", -1)) != expected_hop:
        return f"hop_mismatch:{rec.get('hop_count')}"
    expected_tickers = sorted({c["ticker"] for c in bundle})
    got = sorted(rec.get("tickers", []))
    if set(got) != set(expected_tickers):
        return f"ticker_mismatch:{got}vs{expected_tickers}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts-per-cell", type=int, default=3,
                        help="Retries per target question (for validation failures).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after generating this many rows (smoke test).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI  # deferred
    client = OpenAI()

    chunks = _load_chunks(CHUNKS_PATH)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    cache = Cache(CACHE_DIR)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    generated = 0
    invalid = 0
    per_cell: dict[tuple[str, int], int] = defaultdict(int)
    t0 = time.time()

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for (scope, hop), target in TARGETS.items():
            attempts = 0
            max_attempts = target * args.max_attempts_per_cell
            while per_cell[(scope, hop)] < target and attempts < max_attempts:
                attempts += 1
                bundle = _sample_bundle(chunks, scope, hop, rng)
                if bundle is None:
                    print(f"  [{scope}/h{hop}] no bundle possible; skipping cell",
                          flush=True)
                    break
                if args.dry_run:
                    per_cell[(scope, hop)] += 1
                    continue
                prompt = _render_prompt(template, bundle, scope, hop)
                try:
                    rec = _synth_one(prompt, args.model, client, cache)
                except Exception as exc:
                    print(f"  [{scope}/h{hop}/a{attempts}] API error: {type(exc).__name__}: {str(exc)[:80]}",
                          flush=True)
                    continue
                err = _validate(rec or {}, scope, hop, bundle)
                if err:
                    invalid += 1
                    continue
                qid = f"S{generated+1:04d}"
                row = {
                    "question_id": qid,
                    "question": rec["question"],
                    "answer": rec["answer"],
                    "hop_count": hop,
                    "scope": scope,
                    "tickers": sorted({c["ticker"] for c in bundle}),
                    "years": sorted({c["fy"] for c in bundle}),
                    "source_chunks": [c["chunk_id"] for c in bundle],
                    "reasoning_chain": rec.get("reasoning_chain", []),
                    "leakage_check": rec.get("leakage_check"),
                    "ambiguity_check": rec.get("ambiguity_check"),
                    "synth_model": args.model,
                    "source_dataset": "synth_v1",
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                per_cell[(scope, hop)] += 1
                generated += 1
                if generated % 10 == 0:
                    elapsed = time.time() - t0
                    rate = generated / elapsed if elapsed > 0 else 0
                    print(
                        f"  [gen {generated}] cell {scope}/h{hop} "
                        f"{per_cell[(scope,hop)]}/{target}  "
                        f"invalid={invalid}  rate={rate:.1f}/s",
                        flush=True,
                    )
                if args.limit is not None and generated >= args.limit:
                    break
            if args.limit is not None and generated >= args.limit:
                break

    print()
    print(f"Generated: {generated}")
    print(f"Invalid (rejected by validator): {invalid}")
    for (scope, hop), n in per_cell.items():
        target = TARGETS[(scope, hop)]
        print(f"  {scope}/h{hop}: {n}/{target}")
    print(f"Output: {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
