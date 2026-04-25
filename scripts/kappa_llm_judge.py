"""Compute Cohen's κ between Stage-2 (gpt-4o-mini) labels and a second
LLM judge (gpt-4o), as an LLM-vs-LLM agreement check.

NOT a substitute for human inter-rater reliability — the framing is
explicit in the report's Reliability subsection.

Procedure
---------
1. Stratified sample 30 predictions across A1, A2, B1, NF, A4.
2. For each, call gpt-4o with the same prompt as Stage 2.
3. Parse the second judge's primary_cause.
4. Compute Cohen's κ between (Stage 2, second judge).
5. Append the result to data/eval/failure_taxonomy/report.md
   (replacing the κ-section placeholder).

Output
------
- data/eval/failure_taxonomy/kappa_llm_sample.jsonl (per-row both labels)
- data/eval/failure_taxonomy/report.md (updated Reliability section)
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache
from scripts.classify_failures_llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    _call_with_cache,
    parse_response,
    render_prompt,
    _load_chunks,
)
from scripts.kappa_sample import _interpret, cohen_kappa

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED = REPO_ROOT / "data" / "eval" / "failure_taxonomy" / "classified_predictions.jsonl"
OUT_SAMPLE = REPO_ROOT / "data" / "eval" / "failure_taxonomy" / "kappa_llm_sample.jsonl"
REPORT_PATH = REPO_ROOT / "data" / "eval" / "failure_taxonomy" / "report.md"
PROMPT_PATH = REPO_ROOT / "prompts" / "classify_failure_v1.txt"
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "kappa_judge"

JUDGE_MODEL = "gpt-4o-2024-11-20"
# The Stage 2 prompt only emits A1/A2/B1, so the judge can only choose
# among these three. Sampling rule-labeled NF/A4 would compare
# apples-to-oranges. Restrict to Stage 2-resolved rows only.
TARGETS = ["A1", "A2", "B1"]
N_PER_CATEGORY = 10
SEED = 42


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    rows = [json.loads(l) for l in CLASSIFIED.open() if l.strip()]
    # Only Stage 2 LLM-resolved rows are valid for κ — rule-labeled
    # rows have categories the judge prompt cannot emit.
    rows = [r for r in rows if not r.get("rule_matched")]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    chunks = _load_chunks(CHUNKS_PATH)
    cache = Cache(CACHE_DIR)

    # Stratified sample.
    by_cat: dict[str, list[dict]] = {c: [] for c in TARGETS}
    for r in rows:
        c = r.get("primary_cause")
        if c in by_cat:
            by_cat[c].append(r)
    print(f"Eligible rows (Stage 2 resolved): {len(rows)} "
          f"(A1={len(by_cat['A1'])}, A2={len(by_cat['A2'])}, B1={len(by_cat['B1'])})")
    rng = random.Random(SEED)
    sample: list[dict] = []
    for c in TARGETS:
        pool = by_cat[c]
        if not pool:
            continue
        rng.shuffle(pool)
        sample.extend(pool[:N_PER_CATEGORY])
    print(f"Sampled {len(sample)} rows across {len(TARGETS)} categories.")

    OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    annotated: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(sample, 1):
        prompt = render_prompt(template, row, chunks)
        try:
            raw = _call_with_cache(prompt, client, cache, JUDGE_MODEL,
                                   DEFAULT_TEMP, DEFAULT_MAX_TOKENS)
        except Exception as exc:
            print(f"  [{i}/{len(sample)}] {row['question_id']} FAILED: "
                  f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            continue
        primary, secondary, reason = parse_response(raw)
        annotated.append({
            **row,
            "judge_label": primary,
            "judge_reason": reason,
        })
        if i % 10 == 0 or i == len(sample):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(sample)}] rate={i/elapsed:.1f}/s", flush=True)

    OUT_SAMPLE.write_text(
        "\n".join(json.dumps(r) for r in annotated) + "\n", encoding="utf-8"
    )
    if not annotated:
        raise SystemExit("no rows annotated; cannot compute κ.")

    # Compute κ — only over rows where judge produced a non-null label.
    pairs = [(r["primary_cause"], r["judge_label"]) for r in annotated
             if r["judge_label"] is not None]
    if not pairs:
        raise SystemExit("judge returned no valid labels.")
    stage2 = [a for a, b in pairs]
    judge = [b for a, b in pairs]
    kappa = cohen_kappa(stage2, judge)
    interp = _interpret(kappa)
    n_pairs = len(pairs)
    agreement_pct = 100 * sum(1 for a, b in pairs if a == b) / n_pairs

    print(f"\nLLM-vs-LLM κ (Stage 2 = gpt-4o-mini vs.\\ judge = {JUDGE_MODEL}):")
    print(f"  n={n_pairs}, observed agreement={agreement_pct:.1f}%, κ={kappa:.3f} ({interp})")
    print(f"  judge label distribution: {Counter(judge)}")

    # Update report.md.
    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text(encoding="utf-8")
        block = (
            "## Reliability\n\n"
            f"**LLM-vs-LLM agreement** between Stage 2 (gpt-4o-mini, "
            f"production classifier) and a second judge (gpt-4o) on a "
            f"stratified sample of {n_pairs} rows: "
            f"observed agreement **{agreement_pct:.1f}\\%**, "
            f"Cohen's $\\kappa$ = **{kappa:.3f}** ({interp}).\n\n"
            f"This is _not_ formal human inter-rater reliability. We frame "
            f"it as cross-LLM consistency evidence; the size of $\\kappa$ "
            f"reflects how reproducible the Stage 2 labels are when a "
            f"larger model judges the same prompt.\n\n"
            f"Sample file: [`kappa_llm_sample.jsonl`]({OUT_SAMPLE.name}).\n"
        )
        if "## Reliability" in text:
            start = text.index("## Reliability")
            end = text.find("\n## ", start + 1)
            end = end if end != -1 else len(text)
            text = text[:start] + block + ("\n" + text[end:] if end != len(text) else "\n")
        else:
            text = text.rstrip() + "\n\n" + block
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"Report updated: {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
