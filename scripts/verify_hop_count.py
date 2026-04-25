"""T7: Auto-verify that each synth question truly requires ALL its seed chunks.

Procedure per question
----------------------
1. For each chunk in `source_chunks` (there are N = hop_count), drop it
   and feed the remaining N-1 chunks to gpt-4o-mini with the same answer
   prompt used in evaluation.
2. Score the returned answer against the synthetic gold with Token-F1.
3. A chunk is "essential" if removing it drops F1 below
   `--essential-threshold` (default 0.5). If every chunk is essential,
   the question is marked `hop_verified: true`. Otherwise it carries the
   set of non-essential chunk_ids and is marked `hop_verified: false`.

Input:  data/qa/synth_pool_v1.jsonl
Output: data/qa/synth_pool_verified.jsonl
        — same rows plus {hop_verified, non_essential_chunks, per_drop_f1}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.answer import render_answer_prompt
from src.cache import Cache
from src.eval import f1_token

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
POOL_IN = REPO_ROOT / "data" / "qa" / "synth_pool_v1.jsonl"
POOL_OUT = REPO_ROOT / "data" / "qa" / "synth_pool_verified.jsonl"
ANSWER_PROMPT = REPO_ROOT / "prompts" / "answer_v1.txt"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "verify"

VERIFIER_MODEL = "gpt-4o-mini-2024-07-18"
ESSENTIAL_THRESHOLD = 0.5


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["chunk_id"]] = r
    return out




def _answer(
    prompt: str, client, cache: Cache, model: str, max_tokens: int = 400
) -> str:
    params = {"model": model, "temperature": 0.0, "max_tokens": max_tokens}
    key = cache.key_for(model, prompt, params)
    cached = cache.get(key)
    if cached is not None:
        return cached["answer"]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    ans = resp.choices[0].message.content or ""
    cache.put(key, {"answer": ans, "model": model})
    return ans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=VERIFIER_MODEL)
    parser.add_argument("--essential-threshold", type=float,
                        default=ESSENTIAL_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    chunks = _load_chunks(CHUNKS_PATH)
    template = ANSWER_PROMPT.read_text(encoding="utf-8")
    cache = Cache(CACHE_DIR)
    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with POOL_IN.open("r", encoding="utf-8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    if args.limit is not None:
        qs = qs[: args.limit]

    print(f"Verifying {len(qs)} questions with {args.model} "
          f"(threshold={args.essential_threshold})")

    n_verified = 0
    n_hop_violation = 0
    t0 = time.time()

    with POOL_OUT.open("w", encoding="utf-8") as fout:
        for i, q in enumerate(qs, 1):
            src_cids = q["source_chunks"]
            gold = q["answer"]
            hop = q["hop_count"]
            per_drop: dict[str, float] = {}
            non_essential: list[str] = []
            for dropped_cid in src_cids:
                remaining = [chunks[c] for c in src_cids if c != dropped_cid and c in chunks]
                if len(remaining) != hop - 1:
                    per_drop[dropped_cid] = -1.0
                    continue
                prompt = render_answer_prompt(template, q["question"], remaining)
                try:
                    ans = _answer(prompt, client, cache, args.model)
                except Exception as exc:
                    print(f"  [{i}/{len(qs)}] {q['question_id']} drop={dropped_cid} "
                          f"FAILED: {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                    per_drop[dropped_cid] = -1.0
                    continue
                f1 = f1_token(ans, gold)
                per_drop[dropped_cid] = f1
                if f1 >= args.essential_threshold:
                    non_essential.append(dropped_cid)

            verified = len(non_essential) == 0 and all(v >= 0 for v in per_drop.values())
            if verified:
                n_verified += 1
            else:
                n_hop_violation += 1

            out = dict(q)
            out["hop_verified"] = verified
            out["non_essential_chunks"] = non_essential
            out["per_drop_f1"] = per_drop
            out["verify_model"] = args.model
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 10 == 0 or i == len(qs):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(qs)}] verified={n_verified} violations={n_hop_violation} "
                    f"rate={rate:.1f}/s",
                    flush=True,
                )

    print()
    print(f"Verified (all chunks essential): {n_verified}/{len(qs)} "
          f"({100*n_verified/max(len(qs),1):.1f}%)")
    print(f"Hop violations: {n_hop_violation}/{len(qs)}")
    print(f"Output: {POOL_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
