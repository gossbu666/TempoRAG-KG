"""One-time build of chunk embeddings for retrieval.

Embeds all 7,467 chunks in `data/samples/10k_chunks.jsonl` with OpenAI
`text-embedding-3-small` and saves (embeddings.npy, chunk_ids.json) under
`data/embeddings/chunks/`. The index loads in one `np.load` call from
`src.retrieval.load_index`.

Cost: 7,467 × ~500 tokens × $0.02 / 1M = **~$0.07** one-time.
Idempotent: running again with existing output does nothing unless
`--rebuild` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval import OpenAIEmbeddingClient, embed_chunks, save_index

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
OUT_DIR = REPO_ROOT / "data" / "embeddings" / "chunks"


def _load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small, 1536-dim).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Inputs per API call. 100 balances latency and error isolation.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Embed only the first N chunks (smoke test).",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Overwrite existing index (otherwise refuse if present).",
    )
    args = parser.parse_args()

    load_dotenv()

    if (OUT_DIR / "embeddings.npy").exists() and not args.rebuild:
        print(f"Index already exists at {OUT_DIR}; pass --rebuild to overwrite.")
        return

    chunks = _load_chunks(CHUNKS_PATH)
    if args.limit is not None:
        chunks = chunks[: args.limit]
    print(f"Loading {len(chunks)} chunks from {CHUNKS_PATH.relative_to(REPO_ROOT)}")

    client = OpenAIEmbeddingClient(model_name=args.model)
    print(f"Embedding with model: {args.model}  (batch size {args.batch_size})")

    t0 = time.time()
    embeddings = embed_chunks(chunks, client, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(
        f"Embedded {len(chunks)} chunks in {elapsed:.1f}s "
        f"(shape={embeddings.shape}, dtype={embeddings.dtype})"
    )

    chunk_ids = [c["chunk_id"] for c in chunks]
    save_index(embeddings, chunk_ids, OUT_DIR)
    print(f"Wrote {OUT_DIR.relative_to(REPO_ROOT)}/embeddings.npy + chunk_ids.json")


if __name__ == "__main__":
    main()
