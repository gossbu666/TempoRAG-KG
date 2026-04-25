"""P1 pilot — run 20 stratified 10-K chunks through Gemini 2.0 Flash.

Gates the full KG extraction run (7,467 chunks, ~$0.83). Surfaces prompt bugs
(fiscal-year resolution, forward-looking detection, hallucinated evidence,
schema violations) before we spend the real money.

Strategy:
    - Sample 4 chunks per item (1, 1A, 7, 7A, 8) = 20 total.
    - Prefer ticker diversity inside each item bucket.
    - Seeded from env `RANDOM_SEED` so re-runs hit the same 20 chunks (and
      the raw-response cache).

Writes:
    data/kg/pilot/raw/<chunk_id>.json   — per-chunk record (chunk + result)
    data/kg/pilot/pilot_triples.jsonl   — flattened triples (for quick grep)
    data/kg/pilot/pilot_report.md       — review-ready markdown

The report is the primary artifact for the review gate: it shows per-chunk
stats, the temporal_type distribution, validation failures, and 5 random
chunk samples in full for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache
from src.kg_extract import (
    ExtractionResult,
    GeminiClient,
    extract_triples,
    load_prompt_template,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
PILOT_DIR = REPO_ROOT / "data" / "kg" / "pilot"
RAW_DIR = PILOT_DIR / "raw"
TRIPLES_PATH = PILOT_DIR / "pilot_triples.jsonl"
REPORT_PATH = PILOT_DIR / "pilot_report.md"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "pilot"

PILOT_N_PER_ITEM = 4
TARGET_ITEMS = ("1", "1A", "7", "7A", "8")
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratified_sample(chunks: list[dict], seed: int) -> list[dict]:
    """Pick PILOT_N_PER_ITEM chunks per item, maximizing ticker spread.

    Algorithm: shuffle within item (seeded), then round-robin across ticker
    buckets so we don't pile 4 Item-8 chunks onto one filing.
    """
    rng = random.Random(seed)
    picked: list[dict] = []
    for item in TARGET_ITEMS:
        by_ticker: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            if c["item"] == item:
                by_ticker[c["ticker"]].append(c)
        for tk in by_ticker:
            by_ticker[tk].sort(key=lambda c: c["chunk_id"])
            rng.shuffle(by_ticker[tk])
        tickers = sorted(by_ticker.keys())
        rng.shuffle(tickers)
        bucket: list[dict] = []
        while len(bucket) < PILOT_N_PER_ITEM and tickers:
            for tk in list(tickers):
                if not by_ticker[tk]:
                    tickers.remove(tk)
                    continue
                bucket.append(by_ticker[tk].pop())
                if len(bucket) == PILOT_N_PER_ITEM:
                    break
        picked.extend(bucket)
    picked.sort(key=lambda c: c["chunk_id"])
    return picked


def _dump_per_chunk(chunk: dict, result: ExtractionResult) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{chunk['chunk_id']}.json"
    payload = {
        "chunk_id": chunk["chunk_id"],
        "ticker": chunk["ticker"],
        "fy": chunk["fy"],
        "item": chunk["item"],
        "chunk_text": chunk["text"],
        "model": result.model,
        "cache_hit": result.cache_hit,
        "parse_errors": result.parse_errors,
        "n_triples": len(result.triples),
        "n_dropped": len(result.parse_errors),
        "triples": result.triples,
        "raw_response": result.raw_response,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_flat_triples(results: list[tuple[dict, ExtractionResult]]) -> None:
    with TRIPLES_PATH.open("w", encoding="utf-8") as f:
        for chunk, res in results:
            for t in res.triples:
                row = {**t, "chunk_id": chunk["chunk_id"],
                       "ticker": chunk["ticker"], "fy": chunk["fy"],
                       "item": chunk["item"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(
    results: list[tuple[dict, ExtractionResult]],
    model: str,
    seed: int,
    sample_n_show: int,
) -> None:
    total = len(results)
    # Per-triple validation classes each chunk into one of three buckets:
    #   full    — model returned triples, all passed
    #   partial — model returned triples, some passed and some dropped
    #   fatal   — nothing survived (bad JSON, wrong schema, or all triples dropped)
    n_full = sum(1 for _, r in results if r.triples and not r.parse_errors)
    n_partial = sum(1 for _, r in results if r.triples and r.parse_errors)
    n_fatal = sum(1 for _, r in results if not r.triples)
    n_triples = sum(len(r.triples) for _, r in results)
    n_dropped = sum(len(r.parse_errors) for _, r in results)
    by_item = Counter(c["item"] for c, _ in results)
    by_ticker = Counter(c["ticker"] for c, _ in results)
    temporal_types = Counter(
        t["metadata"].get("temporal_type", "<missing>")
        for _, r in results for t in r.triples
    )
    errors = [
        (c["chunk_id"], err)
        for c, r in results for err in r.parse_errors
    ]
    triples_per_chunk = sorted(
        [(c["chunk_id"], len(r.triples), len(r.parse_errors), r.parse_errors) for c, r in results],
        key=lambda x: (-x[1], x[2]),
    )

    rng = random.Random(seed + 1)  # different seed than sampling so we show variety
    shown = rng.sample(range(len(results)), k=min(sample_n_show, len(results)))
    shown.sort()

    lines = [
        "# P1 Pilot Report",
        "",
        f"**Model:** `{model}`",
        f"**Seed:** {seed}",
        f"**Chunks sampled:** {total} (stratified: {PILOT_N_PER_ITEM} per item × {len(TARGET_ITEMS)} items)",
        f"**Prompt:** [prompts/extract_v1.txt](../../../prompts/extract_v1.txt)",
        "",
        "## Headline",
        "",
        f"- Chunks full-success: **{n_full}/{total}** (all triples passed)",
        f"- Chunks partial-success: **{n_partial}/{total}** (kept valid, dropped some)",
        f"- Chunks fatal: **{n_fatal}/{total}** (nothing kept)",
        f"- Total triples kept: **{n_triples}** (avg {n_triples/max(total,1):.1f} per chunk)",
        f"- Total triples dropped: **{n_dropped}**",
        f"- Cache hits: {sum(1 for _, r in results if r.cache_hit)}/{total}",
        "",
        "## Distributions",
        "",
        "### By item",
        "",
        "| Item | Chunks | Triples |",
        "|---|---:|---:|",
    ]
    for it in TARGET_ITEMS:
        triples_for_item = sum(len(r.triples) for c, r in results if c["item"] == it)
        lines.append(f"| {it} | {by_item[it]} | {triples_for_item} |")
    lines += [
        "",
        "### By ticker (sampled)",
        "",
        "| Ticker | Chunks |",
        "|---|---:|",
    ]
    for tk, n in sorted(by_ticker.items()):
        lines.append(f"| {tk} | {n} |")
    lines += [
        "",
        "### Temporal type (across all triples)",
        "",
        "| temporal_type | Count | % |",
        "|---|---:|---:|",
    ]
    denom = max(n_triples, 1)
    for tt in ("explicit", "relative", "forward_looking", "unknown"):
        n = temporal_types.get(tt, 0)
        lines.append(f"| {tt} | {n} | {100*n/denom:.1f}% |")
    other = {k: v for k, v in temporal_types.items()
             if k not in {"explicit", "relative", "forward_looking", "unknown"}}
    for k, v in other.items():
        lines.append(f"| `{k}` (UNEXPECTED) | {v} | {100*v/denom:.1f}% |")

    lines += [
        "",
        "## Validation failures",
        "",
    ]
    if not errors:
        lines.append("*None — every triple passed schema + hallucination guard.*")
    else:
        lines += ["| chunk_id | error |", "|---|---|"]
        for cid, err in errors:
            lines.append(f"| {cid} | {err} |")

    lines += [
        "",
        "## Triples-per-chunk spread",
        "",
        "| chunk_id | n_kept | n_dropped | first_error |",
        "|---|---:|---:|---|",
    ]
    for cid, n, nd, errs in triples_per_chunk:
        first_err = errs[0] if errs else ""
        lines.append(f"| {cid} | {n} | {nd} | {first_err} |")

    lines += [
        "",
        "## Sample outputs (eyeball these)",
        "",
        f"*{len(shown)} chunks picked at random from the 20. For each, the "
        "first 300 chars of the passage plus all extracted triples. "
        "Open the matching `raw/<chunk_id>.json` for full context.*",
        "",
    ]
    for i in shown:
        chunk, res = results[i]
        preview = chunk["text"].strip().replace("\n", " ")[:300]
        lines += [
            f"### {chunk['chunk_id']}  ({chunk['ticker']} FY{chunk['fy']} Item {chunk['item']})",
            "",
            f"**Passage preview (300 chars):** {preview}…",
            "",
            f"**n_kept:** {len(res.triples)}  |  **n_dropped:** {len(res.parse_errors)}  |  **first_error:** {res.parse_errors[0] if res.parse_errors else 'None'}",
            "",
        ]
        if res.triples:
            lines += ["| subject | predicate | object | valid_from | valid_to | t_type |",
                      "|---|---|---|---|---|---|"]
            for t in res.triples[:8]:
                subj = str(t.get("subject", ""))[:40]
                pred = str(t.get("predicate", ""))[:40]
                obj = str(t.get("object", ""))[:40]
                vf = t.get("valid_from")
                vt = t.get("valid_to")
                tt = t.get("metadata", {}).get("temporal_type", "")
                lines.append(f"| {subj} | {pred} | {obj} | {vf} | {vt} | {tt} |")
            if len(res.triples) > 8:
                lines.append(f"| *…+{len(res.triples)-8} more* | | | | | |")
        else:
            lines.append("*No triples extracted.*")
        lines.append("")

    lines += [
        "## Review checklist",
        "",
        "Before authorizing full extract, eyeball the 5 samples above and",
        "check [docs/prompt_review.md](../../../docs/prompt_review.md) §3. Red flags:",
        "",
        "- `temporal_type` mostly `relative` or `unknown` → prompt rules aren't resolving with chunk metadata",
        "- Fiscal year off by one (MSFT fiscal 2023 ending June 2023 → must be 2023, not 2022)",
        "- Evidence strings that look invented or paraphrased → hallucination guard should have caught them",
        "- Risk-Factor prose producing triples with hypothetical subjects (\"a cyberattack\") → the 'What NOT to extract' rule failed",
        "- Near-zero triples on Item 7 (MD&A) chunks → passage is rich but model is being too conservative",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-show", type=int, default=5, help="Number of chunks to include in-full in the report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the API calls — just show which chunks would be picked")
    args = parser.parse_args()

    load_dotenv()
    seed_env = os.environ.get("RANDOM_SEED")
    seed = int(seed_env) if seed_env else 42

    chunks = _load_chunks(CHUNKS_PATH)
    picked = _stratified_sample(chunks, seed=seed)
    print(f"Picked {len(picked)} chunks (seed={seed}):")
    for c in picked:
        print(f"  {c['chunk_id']}  ({c['token_count']} tok)")

    if args.dry_run:
        return

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    template = load_prompt_template(REPO_ROOT / "prompts" / "extract_v1.txt")
    cache = Cache(CACHE_DIR)
    client = GeminiClient(model_name=args.model)

    results: list[tuple[dict, ExtractionResult]] = []
    for i, chunk in enumerate(picked, 1):
        print(f"[{i:>2}/{len(picked)}] {chunk['chunk_id']}...", end=" ", flush=True)
        res = extract_triples(chunk, template, client, cache, model=args.model)
        _dump_per_chunk(chunk, res)
        results.append((chunk, res))
        status = "CACHE" if res.cache_hit else "LIVE"
        n_dropped = len(res.parse_errors)
        drop_msg = f" dropped={n_dropped}" if n_dropped else ""
        print(f"{status}  kept={len(res.triples)}{drop_msg}")

    _write_flat_triples(results)
    _write_report(results, model=args.model, seed=seed, sample_n_show=args.n_show)

    n_full = sum(1 for _, r in results if r.triples and not r.parse_errors)
    n_partial = sum(1 for _, r in results if r.triples and r.parse_errors)
    n_fatal = sum(1 for _, r in results if not r.triples)
    total_triples = sum(len(r.triples) for _, r in results)
    total_dropped = sum(len(r.parse_errors) for _, r in results)
    print()
    print(f"full/partial/fatal: {n_full}/{n_partial}/{n_fatal} of {len(results)}")
    print(f"kept: {total_triples}  dropped: {total_dropped}")
    print(f"report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"triples: {TRIPLES_PATH.relative_to(REPO_ROOT)}")
    print(f"raw: {RAW_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
