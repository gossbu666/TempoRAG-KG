"""Full KG extraction — run all 7,467 10-K chunks through Gemini 2.5-flash-lite.

Follows the same contract as `run_pilot.py` but over the full corpus.
Resumable: cache hits are free; transient API errors trigger exponential
backoff; KeyboardInterrupt flushes progress to disk before exiting so a
re-run picks up exactly where it left off.

Shares the pilot cache directory on purpose — the 20 pilot chunks are
already cached there with matching (model, template, params), so they
hit for free. Outputs land in `data/kg/full/` so pilot artifacts stay
untouched for the review gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.cache import Cache
from src.kg_extract import (
    ExtractionResult,
    GeminiClient,
    OpenAIClient,
    extract_triples,
    load_prompt_template,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
OUT_DIR = REPO_ROOT / "data" / "kg" / "full"
RAW_DIR = OUT_DIR / "raw"
TRIPLES_PATH = OUT_DIR / "triples.jsonl"
PROGRESS_PATH = OUT_DIR / "progress.jsonl"
REPORT_PATH = OUT_DIR / "report.md"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "pilot"

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4.1-nano",
}
RETRY_MAX = 6
RETRY_BASE_SEC = 1.5


def _load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_done(path: Path) -> set[str]:
    """chunk_ids already written to progress.jsonl — resume skips these."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _extract_with_retry(
    chunk: dict, template: str, client: GeminiClient, cache: Cache, model: str
) -> ExtractionResult:
    last_exc: Exception | None = None
    for attempt in range(RETRY_MAX):
        try:
            return extract_triples(chunk, template, client, cache, model=model)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # rate limit, transient 5xx, network, ...
            last_exc = exc
            if attempt == RETRY_MAX - 1:
                break
            backoff = RETRY_BASE_SEC * (2 ** attempt)
            print(f"    retry {attempt+1}/{RETRY_MAX} after {backoff:.1f}s: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def _dump_per_chunk(chunk: dict, result: ExtractionResult) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{chunk['chunk_id']}.json"
    payload = {
        "chunk_id": chunk["chunk_id"],
        "ticker": chunk["ticker"],
        "fy": chunk["fy"],
        "item": chunk["item"],
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


def _append_progress(chunk: dict, result: ExtractionResult) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "chunk_id": chunk["chunk_id"],
        "ticker": chunk["ticker"],
        "fy": chunk["fy"],
        "item": chunk["item"],
        "cache_hit": result.cache_hit,
        "n_triples": len(result.triples),
        "n_dropped": len(result.parse_errors),
    }
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_triples(chunk: dict, result: ExtractionResult) -> None:
    TRIPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIPLES_PATH.open("a", encoding="utf-8") as f:
        for t in result.triples:
            row = {**t, "chunk_id": chunk["chunk_id"],
                   "ticker": chunk["ticker"], "fy": chunk["fy"],
                   "item": chunk["item"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(model: str, progress: list[dict]) -> None:
    total = len(progress)
    n_full = sum(1 for p in progress if p["n_triples"] > 0 and p["n_dropped"] == 0)
    n_partial = sum(1 for p in progress if p["n_triples"] > 0 and p["n_dropped"] > 0)
    n_fatal = sum(1 for p in progress if p["n_triples"] == 0)
    n_triples = sum(p["n_triples"] for p in progress)
    n_dropped = sum(p["n_dropped"] for p in progress)
    n_cache = sum(1 for p in progress if p["cache_hit"])

    by_item = Counter(p["item"] for p in progress)
    by_ticker = Counter(p["ticker"] for p in progress)
    triples_by_item = Counter()
    triples_by_ticker = Counter()
    for p in progress:
        triples_by_item[p["item"]] += p["n_triples"]
        triples_by_ticker[p["ticker"]] += p["n_triples"]

    lines = [
        "# Full KG Extraction Report",
        "",
        f"**Model:** `{model}`",
        f"**Chunks processed:** {total} / 7467",
        f"**Prompt:** [prompts/extract_v1.txt](../../../prompts/extract_v1.txt)",
        "",
        "## Headline",
        "",
        f"- Chunks full-success: **{n_full}/{total}**",
        f"- Chunks partial-success: **{n_partial}/{total}**",
        f"- Chunks fatal: **{n_fatal}/{total}**",
        f"- Total triples kept: **{n_triples}** (avg {n_triples/max(total,1):.1f} per chunk)",
        f"- Total triples dropped: **{n_dropped}**",
        f"- Cache hits: {n_cache}/{total}",
        "",
        "## By item",
        "",
        "| Item | Chunks | Triples | Avg |",
        "|---|---:|---:|---:|",
    ]
    for it in ("1", "1A", "7", "7A", "8"):
        c = by_item.get(it, 0)
        t = triples_by_item.get(it, 0)
        avg = t / c if c else 0
        lines.append(f"| {it} | {c} | {t} | {avg:.1f} |")
    lines += ["", "## By ticker", "",
              "| Ticker | Chunks | Triples | Avg |",
              "|---|---:|---:|---:|"]
    for tk in sorted(by_ticker):
        c = by_ticker[tk]
        t = triples_by_ticker[tk]
        avg = t / c if c else 0
        lines.append(f"| {tk} | {c} | {t} | {avg:.1f} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODEL_BY_PROVIDER), default="gemini")
    parser.add_argument("--model", default=None,
                        help="Override the default model for the chosen provider.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N remaining chunks (smoke test).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Process in random order (seeded) so partial runs are representative.")
    args = parser.parse_args()
    model = args.model or DEFAULT_MODEL_BY_PROVIDER[args.provider]

    load_dotenv()
    chunks = _load_chunks(CHUNKS_PATH)
    done = _load_done(PROGRESS_PATH)
    remaining = [c for c in chunks if c["chunk_id"] not in done]
    if args.shuffle:
        random.Random(42).shuffle(remaining)
    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(f"Corpus: {len(chunks)}  done: {len(done)}  remaining (this run): {len(remaining)}")
    if not remaining:
        print("Nothing to do — run `_write_report` once more if you want a fresh report.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = load_prompt_template(REPO_ROOT / "prompts" / "extract_v1.txt")
    cache = Cache(CACHE_DIR)
    if args.provider == "openai":
        client = OpenAIClient(model_name=model)
    else:
        client = GeminiClient(model_name=model)
    print(f"Provider: {args.provider}  Model: {model}")

    progress_rows: list[dict] = []
    # Load prior progress so the report reflects the full run, not this slice.
    if PROGRESS_PATH.exists():
        with PROGRESS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        progress_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    stop = {"flag": False}

    def _handle_sigint(signum, frame):  # noqa: ANN001 — signal handler signature
        stop["flag"] = True
        print("\n  SIGINT received — finishing current chunk then exiting.", flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    t0 = time.time()
    n_live = 0
    n_cache = 0
    for i, chunk in enumerate(remaining, 1):
        if stop["flag"]:
            break
        try:
            res = _extract_with_retry(chunk, template, client, cache, model)
        except Exception as exc:
            print(f"[{i}/{len(remaining)}] {chunk['chunk_id']}... FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        _dump_per_chunk(chunk, res)
        _append_progress(chunk, res)
        _append_triples(chunk, res)
        progress_rows.append({
            "chunk_id": chunk["chunk_id"], "ticker": chunk["ticker"],
            "fy": chunk["fy"], "item": chunk["item"],
            "cache_hit": res.cache_hit,
            "n_triples": len(res.triples), "n_dropped": len(res.parse_errors),
        })
        if res.cache_hit:
            n_cache += 1
        else:
            n_live += 1
        if i % 50 == 0 or i == len(remaining):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_sec = (len(remaining) - i) / rate if rate > 0 else 0
            print(f"[{i:>5}/{len(remaining)}] live={n_live} cache={n_cache}  "
                  f"elapsed={elapsed/60:.1f}min  rate={rate:.1f}/s  eta={eta_sec/60:.1f}min",
                  flush=True)

    _write_report(model, progress_rows)
    print()
    print(f"Report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Triples: {TRIPLES_PATH.relative_to(REPO_ROOT)}")
    print(f"Raw: {RAW_DIR.relative_to(REPO_ROOT)}/")
    print(f"Progress: {PROGRESS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
