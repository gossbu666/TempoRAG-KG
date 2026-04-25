"""Stage 2 of the failure-taxonomy classifier: LLM for ambiguous rows.

Reads `rules_stage.jsonl`. Every row with `primary_cause == null` is
sent to `gpt-4o-mini` (temperature 0) with the prompt at
`prompts/classify_failure_v1.txt`. Responses are cached on
`(model, rendered_prompt, temperature=0)` via `src.cache.Cache` so
re-runs are free. Writes the merged `classified_predictions.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache
from src.taxonomy import CATEGORY_CODES

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TEMP = 0.0
DEFAULT_MAX_TOKENS = 200
CACHE_DIR = Path("data/cache/failure_classify")

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_VALID_PRIMARY = {"A1", "A2", "B1"}


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def render_prompt(template: str, row: dict, chunks: dict[str, dict]) -> str:
    """Fill `{QUESTION}`, `{GOLD}`, `{PREDICTION}`, `{CONTEXT}`, `{F1}`."""
    ctx_blocks: list[str] = []
    for cid in row.get("retrieved_ids") or []:
        c = chunks.get(cid)
        if not c:
            continue
        header = f"    [{cid} | {c.get('ticker','?')} FY{c.get('fy','?')} Item {c.get('item','?')}]"
        ctx_blocks.append(header + "\n    " + c.get("text", "").replace("\n", " ")[:900])
    context = "\n".join(ctx_blocks) if ctx_blocks else "    (no chunks retrieved)"
    gold = row.get("gold")
    if isinstance(gold, list):
        gold = " | ".join(str(g) for g in gold)
    f1 = row.get("f1")
    return (template
            .replace("{QUESTION}", str(row.get("question", "")))
            .replace("{GOLD}", str(gold or ""))
            .replace("{PREDICTION}", str(row.get("prediction", "")))
            .replace("{CONTEXT}", context)
            .replace("{F1}", f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1)))


def parse_response(raw: str) -> tuple[str | None, str | None, str]:
    """Return (primary, secondary, reason) or (None, None, raw) on failure."""
    m = _FENCE_RE.match(raw.strip())
    text = m.group(1).strip() if m else raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, None, raw
    if not isinstance(obj, dict):
        return None, None, raw
    primary = obj.get("primary")
    secondary = obj.get("secondary")
    reason = str(obj.get("reason", "")).strip()
    if primary not in _VALID_PRIMARY:
        return None, None, reason or raw
    if secondary is not None and secondary not in CATEGORY_CODES:
        secondary = None
    return primary, secondary, reason


def _call_with_cache(prompt: str, client, cache: Cache, model: str,
                     temperature: float, max_tokens: int) -> str:
    params = {"temperature": temperature, "max_tokens": max_tokens}
    key = cache.key_for(model, prompt, params)
    cached = cache.get(key)
    if cached is not None:
        return cached["response"]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    cache.put(key, {"response": raw, "model": model})
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-jsonl",
                        default="data/eval/failure_taxonomy/rules_stage.jsonl")
    parser.add_argument("--chunks",
                        default="data/samples/10k_chunks.jsonl")
    parser.add_argument("--prompt",
                        default="prompts/classify_failure_v1.txt")
    parser.add_argument("--out",
                        default="data/eval/failure_taxonomy/classified_predictions.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    template = Path(args.prompt).read_text(encoding="utf-8")
    chunks = _load_chunks(Path(args.chunks))
    cache = Cache(CACHE_DIR)

    rows: list[dict] = []
    with open(args.rules_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ambiguous = [r for r in rows if r.get("primary_cause") is None]
    if args.limit is not None:
        ambiguous = ambiguous[: args.limit]
    print(f"Total rows: {len(rows)}  ambiguous (Stage 2): {len(ambiguous)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    resolved: dict[tuple[str, str, str], dict] = {}
    t0 = time.time()
    for i, row in enumerate(ambiguous, 1):
        prompt = render_prompt(template, row, chunks)
        try:
            raw = _call_with_cache(prompt, client, cache,
                                   args.model, DEFAULT_TEMP, DEFAULT_MAX_TOKENS)
        except Exception as exc:
            print(f"  [{i}/{len(ambiguous)}] {row['question_id']} FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            continue
        primary, secondary, reason = parse_response(raw)
        key = (str(row["question_id"]), row["condition"], row["model"])
        resolved[key] = {"primary_cause": primary,
                         "secondary_cause": secondary,
                         "reason": reason}
        if i % 50 == 0 or i == len(ambiguous):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(ambiguous)}] rate={rate:.1f}/s",
                  flush=True)

    # Merge resolutions back into the full row set and write out.
    with open(args.out, "w", encoding="utf-8") as fout:
        for row in rows:
            key = (str(row["question_id"]), row["condition"], row["model"])
            if key in resolved:
                row = {**row, **resolved[key]}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
