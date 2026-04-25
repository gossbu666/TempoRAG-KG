"""Embedding-based retrieval over 10-K chunks.

One-time index build: embed all 7,467 chunks with OpenAI
`text-embedding-3-small`, save as (embeddings.npy, chunk_ids.json). At
query time, embed the question and return top-K chunks by cosine
similarity.

Why not FAISS: 7,467 chunks × 1536 dims = ~45 MB float32. A numpy
dot-product over this runs in <10 ms. FAISS adds a dependency for zero
measurable win at this scale.

Query embeddings are cached via the existing `src.cache.Cache` — re-running
the eval is free.

Four retrieval conditions are implemented here:
  L0 Vanilla      : `retrieve`                    cosine top-k
  L1 TimeFilter   : `retrieve_with_year_filter`   cosine + hard mask on chunk.fy
  L2 KG²RAG       : `retrieve_kg2rag`             cosine seeds + 1-hop KG expansion
  L3 TempoRAG-KG  : `retrieve_temporag_kg`        L2 + triple-level temporal filter
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.cache import Cache


class EmbeddingClient(Protocol):
    """`.embed_many(texts) -> np.ndarray` of shape (len(texts), dim)."""

    def embed_many(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class ChunkIndex:
    """Loaded chunk embeddings + parallel metadata.

    `embeddings` is (N, D) **L2-normalized** so cosine == dot-product.
    `chunk_ids[i]` corresponds to `embeddings[i]`.
    `chunks[chunk_id]` holds the full chunk record (text, ticker, fy, ...).
    """

    embeddings: np.ndarray
    chunk_ids: list[str]
    chunks: dict[str, dict]


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return v / norm


def embed_chunks(
    chunks: list[dict],
    client: EmbeddingClient,
    *,
    batch_size: int = 100,
    text_field: str = "text",
) -> np.ndarray:
    """Embed `chunks[i][text_field]` for every chunk; return (N, D) L2-normalized.

    Batches `batch_size` at a time to stay under OpenAI's per-request limit
    (2048 inputs, but smaller batches give better error isolation).
    """
    vecs_batches: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c[text_field] for c in batch]
        vecs_batches.append(client.embed_many(texts))
    stacked = np.vstack(vecs_batches).astype(np.float32)
    return _normalize(stacked)


def save_index(embeddings: np.ndarray, chunk_ids: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids), encoding="utf-8"
    )


def load_index(
    index_dir: Path,
    chunks_path: Path,
) -> ChunkIndex:
    """Load embeddings.npy + chunk_ids.json, align against chunks JSONL."""
    embeddings = np.load(index_dir / "embeddings.npy")
    chunk_ids = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    chunks: dict[str, dict] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks[rec["chunk_id"]] = rec
    missing = [cid for cid in chunk_ids if cid not in chunks]
    if missing:
        raise ValueError(
            f"{len(missing)} chunk_ids in index but not in {chunks_path}: "
            f"{missing[:3]}..."
        )
    return ChunkIndex(embeddings=embeddings, chunk_ids=chunk_ids, chunks=chunks)


def retrieve(
    question: str,
    index: ChunkIndex,
    client: EmbeddingClient,
    cache: Cache,
    *,
    k: int = 5,
    embed_model: str = "text-embedding-3-small",
) -> list[dict]:
    """Return the top-K chunks most similar to `question` by cosine.

    Query embedding is cached — the same question re-embedded costs nothing.
    Each returned dict is a copy of the chunk record with `retrieval_score`
    added (float in [-1, 1]).
    """
    key = cache.key_for(embed_model, question, {})
    cached = cache.get(key)
    if cached is not None:
        q_vec = np.asarray(cached["embedding"], dtype=np.float32)
    else:
        q_vec = client.embed_many([question])[0].astype(np.float32)
        cache.put(key, {"embedding": q_vec.tolist(), "model": embed_model})
    q_vec = _normalize(q_vec[np.newaxis, :])[0]

    scores = index.embeddings @ q_vec  # (N,)
    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        top_idx = part[np.argsort(-scores[part])]

    results: list[dict] = []
    for idx in top_idx:
        cid = index.chunk_ids[idx]
        rec = dict(index.chunks[cid])
        rec["retrieval_score"] = float(scores[idx])
        results.append(rec)
    return results


def retrieve_with_year_filter(
    question: str,
    years: list[int],
    index: ChunkIndex,
    client: EmbeddingClient,
    cache: Cache,
    *,
    k: int = 5,
    embed_model: str = "text-embedding-3-small",
) -> list[dict]:
    """TimeFilter RAG: vanilla cosine + hard mask on `chunk.fy ∈ years`.

    Fills the ablation cell "temporal filter without KG" — isolates whether
    year-scoped retrieval alone explains the lift, independent of KG structure.
    Falls back to unfiltered retrieval if `years` is empty or the mask
    produces fewer than k candidates (avoids K-variance confounding the
    comparison against vanilla).
    """
    key = cache.key_for(embed_model, question, {})
    cached = cache.get(key)
    if cached is not None:
        q_vec = np.asarray(cached["embedding"], dtype=np.float32)
    else:
        q_vec = client.embed_many([question])[0].astype(np.float32)
        cache.put(key, {"embedding": q_vec.tolist(), "model": embed_model})
    q_vec = _normalize(q_vec[np.newaxis, :])[0]

    scores = index.embeddings @ q_vec

    if years:
        year_set = set(years)
        mask = np.array(
            [index.chunks[cid].get("fy") in year_set for cid in index.chunk_ids],
            dtype=bool,
        )
        if mask.sum() >= k:
            scores = np.where(mask, scores, -np.inf)

    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        top_idx = part[np.argsort(-scores[part])]

    results: list[dict] = []
    for idx in top_idx:
        cid = index.chunk_ids[idx]
        rec = dict(index.chunks[cid])
        rec["retrieval_score"] = float(scores[idx])
        results.append(rec)
    return results


# -----------------------------------------------------------------------------
# KG-augmented retrieval (L2 KG²RAG and L3 TempoRAG-KG)
# -----------------------------------------------------------------------------


@dataclass
class KGIndex:
    """Two lookup maps built from the filtered triples file.

    `chunk_to_triples[chunk_id]` — list of triples originating from that chunk.
    `entity_to_chunks[entity]` — set of chunk_ids where that entity appears as
    subject or object of any triple.

    Entities are normalized via `_normalize_entity` (strip + lowercase) so
    variants like "Apple Inc." / "Apple Inc" / "APPLE INC." collide.
    """

    chunk_to_triples: dict[str, list[dict]]
    entity_to_chunks: dict[str, set[str]]


def _normalize_entity(e: str) -> str:
    return e.strip().lower()


def load_kg_index(triples_path: Path) -> KGIndex:
    """Read the filtered triples JSONL and build chunk/entity indices."""
    chunk_to_triples: dict[str, list[dict]] = defaultdict(list)
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            cid = t.get("chunk_id")
            if not cid:
                continue
            chunk_to_triples[cid].append(t)
            s = _normalize_entity(str(t.get("subject", "")))
            o = _normalize_entity(str(t.get("object", "")))
            if s:
                entity_to_chunks[s].add(cid)
            if o:
                entity_to_chunks[o].add(cid)
    return KGIndex(
        chunk_to_triples=dict(chunk_to_triples),
        entity_to_chunks=dict(entity_to_chunks),
    )


def _get_query_vec(
    question: str,
    client: EmbeddingClient,
    cache: Cache,
    embed_model: str,
) -> np.ndarray:
    """Return the L2-normalized query embedding, using cache when available."""
    key = cache.key_for(embed_model, question, {})
    cached = cache.get(key)
    if cached is not None:
        q_vec = np.asarray(cached["embedding"], dtype=np.float32)
    else:
        q_vec = client.embed_many([question])[0].astype(np.float32)
        cache.put(key, {"embedding": q_vec.tolist(), "model": embed_model})
    return _normalize(q_vec[np.newaxis, :])[0]


def _triple_overlaps_years(t: dict, year_set: set[int]) -> bool:
    """True if the triple's temporal validity intersects `year_set`.

    Rules:
      - If both valid_from and valid_to are null, fall back to chunk.fy
        (captured via the `fy` field on the triple record at write time).
      - If valid_from is a date/year-like string, extract the year and
        compare against the [min, max] of year_set as an interval.
      - Missing valid_to means open-ended (treat as +inf).
    """

    def _yr(v) -> int | None:
        if v is None:
            return None
        s = str(v)
        # Take first 4 digits as year (covers "2021", "2021-12-31", etc.).
        for i in range(len(s) - 3):
            if s[i : i + 4].isdigit():
                return int(s[i : i + 4])
        return None

    if not year_set:
        return True
    lo = min(year_set)
    hi = max(year_set)

    vf = _yr(t.get("valid_from"))
    vt = _yr(t.get("valid_to"))

    if vf is None and vt is None:
        fy = t.get("fy")
        return fy in year_set if fy is not None else False

    vf = vf if vf is not None else -10_000
    vt = vt if vt is not None else 10_000
    return vf <= hi and vt >= lo


def _topk_from_scores(
    scores: np.ndarray, k: int, chunk_ids: list[str], chunks: dict[str, dict]
) -> list[dict]:
    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, k)[:k]
        top_idx = part[np.argsort(-scores[part])]
    results: list[dict] = []
    for idx in top_idx[:k]:
        if not np.isfinite(scores[idx]):
            continue
        cid = chunk_ids[idx]
        rec = dict(chunks[cid])
        rec["retrieval_score"] = float(scores[idx])
        results.append(rec)
    return results


def retrieve_kg2rag(
    question: str,
    index: ChunkIndex,
    kg_index: KGIndex,
    client: EmbeddingClient,
    cache: Cache,
    *,
    k: int = 5,
    seed_k: int = 3,
    embed_model: str = "text-embedding-3-small",
) -> list[dict]:
    """L2 KG²RAG: cosine seeds + 1-hop entity expansion, no temporal filter.

    Isolates the graph-walk effect on top of vanilla cosine retrieval.
    1) Take top `seed_k` chunks by cosine.
    2) Collect every entity (subject+object) appearing in those seed chunks'
       triples.
    3) Expand the candidate pool to every chunk that shares any such entity.
    4) Re-rank seeds ∪ expanded by cosine against the same query, return top-k.

    If the expanded pool is smaller than `k` (graph is too sparse for this
    question) fall back to vanilla cosine top-k so result length is stable.
    """
    q_vec = _get_query_vec(question, client, cache, embed_model)
    scores = index.embeddings @ q_vec  # (N,)

    if seed_k >= len(scores):
        seed_order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, seed_k)[:seed_k]
        seed_order = part[np.argsort(-scores[part])]
    seed_cids = {index.chunk_ids[i] for i in seed_order[:seed_k]}

    # Collect every subject and object entity from triples on the seed
    # chunks. We deliberately keep this expansion broad — including
    # generic entities such as "the Company" or "fiscal year" — to
    # mirror the canonical KG²RAG specification. The Discussion
    # section (§6.1) shows that this breadth is the proximate cause
    # of L2's regression vs. L0: an entity-related chunk is not the
    # same as a question-related chunk, and cosine re-ranking on the
    # diluted pool ends up ranking weaker matches above the original
    # cosine top-k. A future iteration could either filter entities by
    # IDF (drop generic ones) or learn a re-ranker.
    entities: set[str] = set()
    for cid in seed_cids:
        for t in kg_index.chunk_to_triples.get(cid, []):
            s = _normalize_entity(str(t.get("subject", "")))
            o = _normalize_entity(str(t.get("object", "")))
            if s:
                entities.add(s)
            if o:
                entities.add(o)

    expanded_cids: set[str] = set()
    for e in entities:
        expanded_cids.update(kg_index.entity_to_chunks.get(e, ()))

    candidate_cids = seed_cids | expanded_cids
    if len(candidate_cids) < k:
        # Sparse-graph fallback: if the union of cosine seeds and KG
        # expansion still has fewer than k chunks, hand the query back
        # to the vanilla retriever. Keeps the result list length stable
        # so downstream eval doesn't have to handle short contexts.
        return retrieve(
            question, index, client, cache, k=k, embed_model=embed_model
        )

    cid_to_idx = {cid: i for i, cid in enumerate(index.chunk_ids)}
    mask = np.zeros(len(scores), dtype=bool)
    for cid in candidate_cids:
        i = cid_to_idx.get(cid)
        if i is not None:
            mask[i] = True
    masked_scores = np.where(mask, scores, -np.inf)
    return _topk_from_scores(masked_scores, k, index.chunk_ids, index.chunks)


def retrieve_temporag_kg(
    question: str,
    years: list[int],
    index: ChunkIndex,
    kg_index: KGIndex,
    client: EmbeddingClient,
    cache: Cache,
    *,
    k: int = 5,
    seed_k: int = 3,
    embed_model: str = "text-embedding-3-small",
) -> list[dict]:
    """L3 TempoRAG-KG: L2 expansion, but only through triples whose validity
    overlaps `years`.

    Distinct from L1 (which masks chunks by `chunk.fy`) — L3 filters at the
    *triple* level. A chunk from FY2023 that contains a triple with
    valid_from=2019..valid_to=2021 stays reachable for an FY2020 query even
    though the chunk's fy wouldn't pass L1's mask; conversely, triples with
    bounds outside the query years are ignored as expansion paths.

    If `years` is empty, behaves identically to `retrieve_kg2rag`. If the
    temporally-filtered expansion pool is smaller than `k`, fall back to L2
    so result length stays stable.
    """
    if not years:
        return retrieve_kg2rag(
            question, index, kg_index, client, cache,
            k=k, seed_k=seed_k, embed_model=embed_model,
        )

    year_set = set(years)
    q_vec = _get_query_vec(question, client, cache, embed_model)
    scores = index.embeddings @ q_vec

    if seed_k >= len(scores):
        seed_order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, seed_k)[:seed_k]
        seed_order = part[np.argsort(-scores[part])]
    seed_cids = {index.chunk_ids[i] for i in seed_order[:seed_k]}

    entities: set[str] = set()
    for cid in seed_cids:
        for t in kg_index.chunk_to_triples.get(cid, []):
            if not _triple_overlaps_years(t, year_set):
                continue
            s = _normalize_entity(str(t.get("subject", "")))
            o = _normalize_entity(str(t.get("object", "")))
            if s:
                entities.add(s)
            if o:
                entities.add(o)

    expanded_cids: set[str] = set()
    for e in entities:
        for cid in kg_index.entity_to_chunks.get(e, ()):
            for t in kg_index.chunk_to_triples.get(cid, []):
                if _triple_overlaps_years(t, year_set):
                    expanded_cids.add(cid)
                    break

    candidate_cids = seed_cids | expanded_cids
    if len(candidate_cids) < k:
        return retrieve_kg2rag(
            question, index, kg_index, client, cache,
            k=k, seed_k=seed_k, embed_model=embed_model,
        )

    cid_to_idx = {cid: i for i, cid in enumerate(index.chunk_ids)}
    mask = np.zeros(len(scores), dtype=bool)
    for cid in candidate_cids:
        i = cid_to_idx.get(cid)
        if i is not None:
            mask[i] = True
    masked_scores = np.where(mask, scores, -np.inf)
    return _topk_from_scores(masked_scores, k, index.chunk_ids, index.chunks)


# -----------------------------------------------------------------------------
# Concrete OpenAI embedding adapter
# -----------------------------------------------------------------------------

class OpenAIEmbeddingClient:
    """Adapter around OpenAI's `embeddings.create`.

    Default: `text-embedding-3-small` — 1536 dims, $0.02/1M tokens.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI  # deferred
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — refuse to run without a key"
            )
        self._model_name = model_name
        self._client = OpenAI(api_key=resolved_key)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return np.array([e.embedding for e in resp.data], dtype=np.float32)
