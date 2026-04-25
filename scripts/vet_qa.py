"""T8: Human-review CLI for synth QA questions.

Round-robin assigns questions (by hash of question_id mod N_REVIEWERS) to
reviewers so work is split evenly without central coordination. Each
reviewer runs `python scripts/vet_qa.py <name>` and sees only their slice.

Per-question display
--------------------
- Question id, scope, hop, tickers, years.
- Auto-verification outcome (hop_verified + which chunks were non-essential).
- The question text.
- Proposed answer.
- Reasoning chain.
- Source chunks summary (first 250 chars of each).

Decision keys
-------------
  a = accept
  r = reject   → follow-up asks for reason code:
                  1=wrong hop  2=wrong answer  3=wrong scope  4=leakage  5=other
  s = skip for now (re-queued next run)
  v = view full chunk text
  q = quit

Each decision is appended to `data/qa/vet_log.jsonl` immediately — the
session is safe to interrupt at any time.

Final accepted pool (when vetting is done across the team) is derived by
reading the log and keeping rows with the latest `decision=accept`; see
`scripts/finalize_synth_pool.py` (not yet written) for that step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_IN = REPO_ROOT / "data" / "qa" / "synth_pool_verified.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
LOG_PATH = REPO_ROOT / "data" / "qa" / "vet_log.jsonl"

REJECT_REASONS = {
    "1": "wrong_hop",
    "2": "wrong_answer",
    "3": "wrong_scope",
    "4": "leakage",
    "5": "other",
}


def _load_chunks(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["chunk_id"]] = r
    return out


def _load_pool(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _done_qids(log_path: Path, reviewer: str) -> set[str]:
    if not log_path.exists():
        return set()
    out: set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("reviewer") == reviewer and r.get("decision") in ("accept", "reject"):
                out.add(r["question_id"])
    return out


def _hash_bucket(qid: str, n: int) -> int:
    h = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return int(h, 16) % n


def _fmt_question(q: dict, chunks: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("═" * 78)
    lines.append(f"  {q['question_id']}  scope={q['scope']}  hop={q['hop_count']}")
    lines.append(f"  tickers={q.get('tickers')}  years={q.get('years')}")
    lines.append("─" * 78)
    lines.append(f"Q: {q['question']}")
    lines.append("")
    lines.append(f"A: {q['answer']}")
    lines.append("")
    hv = q.get("hop_verified")
    ne = q.get("non_essential_chunks") or []
    if hv is True and not ne:
        lines.append("Auto-verify: ✓ passed — every chunk is essential")
    else:
        lines.append(f"Auto-verify: ✗ non-essential chunks: {ne}")
        per = q.get("per_drop_f1", {})
        for cid, f1 in per.items():
            tag = "❌" if cid in ne else "✓"
            lines.append(f"   {tag} drop({cid}) → F1 {f1:.2f}")
    lines.append("")
    chain = q.get("reasoning_chain") or []
    if chain:
        lines.append("Reasoning chain:")
        for step in chain:
            lines.append(f"  • {step}")
        lines.append("")
    lines.append("Source chunks:")
    for cid in q.get("source_chunks", []):
        c = chunks.get(cid, {})
        preview = (c.get("text") or "")[:250].replace("\n", " ")
        lines.append(f"  [{cid}] ({c.get('ticker','?')} FY{c.get('fy','?')} item {c.get('item','?')})")
        lines.append(f"    {preview}{'...' if c.get('text','') and len(c['text']) > 250 else ''}")
    lines.append("─" * 78)
    return "\n".join(lines)


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewer", help="Your name tag, e.g. 'supanut'.")
    parser.add_argument("--n-reviewers", type=int, default=4,
                        help="Total reviewers in the team; used to pick your slice.")
    parser.add_argument("--bucket", type=int, default=None,
                        help="Override assigned bucket (0..n-reviewers-1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Review at most N questions this run.")
    args = parser.parse_args()

    if not POOL_IN.exists():
        raise SystemExit(f"pool not found at {POOL_IN}; run verify_hop_count.py first.")

    chunks = _load_chunks(CHUNKS_PATH)
    pool = _load_pool(POOL_IN)
    done = _done_qids(LOG_PATH, args.reviewer)

    if args.bucket is not None:
        my_bucket = args.bucket
    else:
        name_hash = int(hashlib.sha256(args.reviewer.encode()).hexdigest(), 16)
        my_bucket = name_hash % args.n_reviewers

    queue = [
        q for q in pool
        if _hash_bucket(q["question_id"], args.n_reviewers) == my_bucket
        and q["question_id"] not in done
    ]
    print(f"Reviewer: {args.reviewer}  bucket: {my_bucket}/{args.n_reviewers}")
    print(f"Pool total: {len(pool)}  your slice: "
          f"{sum(1 for q in pool if _hash_bucket(q['question_id'], args.n_reviewers) == my_bucket)}")
    print(f"Already done by you: {len(done)}")
    print(f"Queue now: {len(queue)}")
    if args.limit:
        queue = queue[: args.limit]

    reviewed = 0
    accepted = 0
    rejected = 0
    try:
        for i, q in enumerate(queue, 1):
            print()
            print(f"═══ Q {i}/{len(queue)} ═══")
            print(_fmt_question(q, chunks))
            while True:
                choice = input("Decision [a=accept, r=reject, s=skip, v=view full, q=quit]: ").strip().lower()
                if choice == "v":
                    for cid in q.get("source_chunks", []):
                        c = chunks.get(cid, {})
                        print(f"\n─── Full chunk {cid} ───")
                        print(c.get("text", "(missing)"))
                        print(f"─── end {cid} ───\n")
                    continue
                if choice in ("a", "r", "s", "q"):
                    break
                print("  ? unknown choice; try again.")
            if choice == "q":
                print("Quitting.")
                break
            if choice == "s":
                _log({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "reviewer": args.reviewer,
                    "question_id": q["question_id"],
                    "decision": "skip",
                })
                continue
            if choice == "a":
                _log({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "reviewer": args.reviewer,
                    "question_id": q["question_id"],
                    "decision": "accept",
                })
                accepted += 1
                reviewed += 1
                continue
            # reject
            while True:
                code = input("Reason (1=wrong_hop 2=wrong_answer 3=wrong_scope 4=leakage 5=other): ").strip()
                if code in REJECT_REASONS:
                    break
                print("  ? unknown code; try again.")
            note = input("Optional note (press Enter to skip): ").strip()
            _log({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "reviewer": args.reviewer,
                "question_id": q["question_id"],
                "decision": "reject",
                "reason": REJECT_REASONS[code],
                "note": note or None,
            })
            rejected += 1
            reviewed += 1
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — log saved, resume anytime.", flush=True)

    print()
    print(f"Reviewed this session: {reviewed}  accept={accepted}  reject={rejected}")
    print(f"Log: {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
