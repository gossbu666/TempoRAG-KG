"""Stricter LLM auto-vet of the synth pool (replaces team human vet).

Reads `data/qa/synth_pool_verified.jsonl` (128 questions; 34 with
`hop_verified=True` after the N-1 chunk verification pass) and asks
gpt-4o to check four axes per question:

  - hop_correct        : does answering really require all source chunks?
  - answer_correct     : is the proposed answer factually consistent
                         with the chunks?
  - scope_correct      : does the labeled scope match the question?
  - leakage_free       : does the question avoid revealing the answer?

Each axis returns true/false plus a one-sentence rationale. A question
is **accepted** only if all four axes pass *and* the prior automatic
hop-verify also passed (`hop_verified=True`).

Output: `data/qa/synth_pool_accepted.jsonl` — final hand-vet-equivalent
pool ready for re-evaluation through the L0/L1/L2/L3 retrievers.

Designed to be a faithful replacement for the four-person team vet
(`scripts/vet_qa.py`), with an explicit caveat that human spot-check
is preferred when time allows.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_IN = REPO_ROOT / "data" / "qa" / "synth_pool_verified.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
ACCEPTED = REPO_ROOT / "data" / "qa" / "synth_pool_accepted.jsonl"
REJECTED = REPO_ROOT / "data" / "qa" / "synth_pool_rejected.jsonl"
REPORT = REPO_ROOT / "docs" / "synth_auto_vet_report.md"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "auto_vet"

JUDGE_MODEL = "gpt-4o-2024-11-20"
TEMP = 0.0
MAX_TOKENS = 500

PROMPT = """You are auditing a synthetic question-answering item. The item
came from a corpus of SEC 10-K filings. Inspect the question, proposed
answer, and source chunks, then judge it on four axes.

Question:    {QUESTION}
Answer:      {ANSWER}
Hop count:   {HOP}
Scope label: {SCOPE}
Tickers:     {TICKERS}
Years:       {YEARS}

Source chunks (each chunk's text was retrieved at synth time):
{CHUNKS}

Decide each axis. Be strict — false positives are worse than false
negatives because rejected items are cheap to drop.

  hop_correct: Does answering the question genuinely require all
    {HOP} source chunks? (false if any single chunk alone could
    answer.)
  answer_correct: Is the proposed answer factually consistent with the
    chunks? (false if any number / entity / claim contradicts the
    chunks, or invents detail not present.)
  scope_correct: Does the labeled scope ({SCOPE}) actually describe the
    question? (e.g. cross_company must reference $\\geq$2 distinct issuers.)
  leakage_free: Does the question avoid revealing the answer in its
    own text?

Respond with ONLY JSON (no fences, no prose):
{{
  "hop_correct":    {{"verdict": true|false, "reason": "..."}},
  "answer_correct": {{"verdict": true|false, "reason": "..."}},
  "scope_correct":  {{"verdict": true|false, "reason": "..."}},
  "leakage_free":   {{"verdict": true|false, "reason": "..."}}
}}
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def render(template: str, q: dict, chunks_map: dict[str, dict]) -> str:
    blocks: list[str] = []
    for cid in q.get("source_chunks") or []:
        c = chunks_map.get(cid)
        if not c:
            continue
        header = f"  [{cid} | {c.get('ticker','?')} FY{c.get('fy','?')} item {c.get('item','?')}]"
        text = c.get("text", "").replace("\n", " ")[:1100]
        blocks.append(f"{header}\n  {text}")
    return (template
            .replace("{QUESTION}", str(q.get("question", "")))
            .replace("{ANSWER}",   str(q.get("answer", "")))
            .replace("{HOP}",      str(q.get("hop_count", 0)))
            .replace("{SCOPE}",    str(q.get("scope", "")))
            .replace("{TICKERS}",  json.dumps(q.get("tickers") or []))
            .replace("{YEARS}",    json.dumps(q.get("years") or []))
            .replace("{CHUNKS}",   "\n\n".join(blocks) if blocks else "  (none)"))


def call(prompt: str, client, cache: Cache) -> str:
    params = {"temperature": TEMP, "max_tokens": MAX_TOKENS}
    key = cache.key_for(JUDGE_MODEL, prompt, params)
    cached = cache.get(key)
    if cached is not None:
        return cached["response"]
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP, max_tokens=MAX_TOKENS,
    )
    raw = resp.choices[0].message.content or ""
    cache.put(key, {"response": raw, "model": JUDGE_MODEL})
    return raw


def parse(raw: str) -> dict | None:
    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


AXES = ("hop_correct", "answer_correct", "scope_correct", "leakage_free")


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    chunks = _load_chunks(CHUNKS_PATH)
    cache = Cache(CACHE_DIR)
    rows = [json.loads(l) for l in POOL_IN.open() if l.strip()]
    print(f"Auto-vetting {len(rows)} synth items with {JUDGE_MODEL}.")

    accepted: list[dict] = []
    rejected: list[dict] = []
    failure_counter: Counter = Counter()
    t0 = time.time()
    for i, q in enumerate(rows, 1):
        prompt = render(PROMPT, q, chunks)
        try:
            raw = call(prompt, client, cache)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] {q['question_id']} ERROR: "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            rejected.append({**q, "vet_error": f"{type(exc).__name__}: {exc}"})
            continue
        verdicts = parse(raw) or {}
        per_axis = {axis: bool(verdicts.get(axis, {}).get("verdict")) for axis in AXES}
        all_pass = all(per_axis.values())
        prior_hop = bool(q.get("hop_verified"))

        record = {
            **q,
            "auto_vet_verdicts": per_axis,
            "auto_vet_reasons": {
                axis: verdicts.get(axis, {}).get("reason", "") for axis in AXES
            },
            "auto_vet_pass": all_pass and prior_hop,
        }
        if all_pass and prior_hop:
            accepted.append(record)
        else:
            for axis in AXES:
                if not per_axis[axis]:
                    failure_counter[axis] += 1
            if not prior_hop:
                failure_counter["prior_hop_unverified"] += 1
            rejected.append(record)

        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(rows)}] accepted={len(accepted)} "
                  f"rejected={len(rejected)} rate={i/elapsed:.1f}/s",
                  flush=True)

    ACCEPTED.parent.mkdir(parents=True, exist_ok=True)
    ACCEPTED.write_text("\n".join(json.dumps(r) for r in accepted) + "\n",
                        encoding="utf-8")
    REJECTED.write_text("\n".join(json.dumps(r) for r in rejected) + "\n",
                        encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synth pool auto-vet report",
        "",
        f"Judge model: `{JUDGE_MODEL}` (temperature {TEMP}).",
        "",
        f"Input: `{POOL_IN.relative_to(REPO_ROOT)}` ({len(rows)} rows).",
        "",
        f"**Accepted: {len(accepted)} / {len(rows)}**  "
        f"({100*len(accepted)/max(len(rows),1):.1f}%).",
        f"Rejected: {len(rejected)}.",
        "",
        "## Rejection breakdown by failed axis",
        "",
        "| Axis | Rejected count |",
        "|---|---:|",
    ]
    for axis, n in failure_counter.most_common():
        lines.append(f"| {axis} | {n} |")
    lines += ["",
              f"Output files: `{ACCEPTED.relative_to(REPO_ROOT)}`, "
              f"`{REJECTED.relative_to(REPO_ROOT)}`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT.relative_to(REPO_ROOT)}")
    print(f"Accepted pool: {ACCEPTED.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
