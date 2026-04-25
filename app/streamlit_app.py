"""TempoRAG-KG demo UI — single-page Streamlit app.

Run:
    PYTHONPATH=. streamlit run app/streamlit_app.py

Shows the full 2×2 ablation (L0 / L1 / L2 / L3) for any typed question over
the 25 × 10-K corpus. For each condition the app displays:

- The top-k retrieved chunks with cosine scores (and a flag marking which
  chunks came from the KG-expanded pool in L2/L3).
- The model's answer.
- If the question matches one of the 129 labeled QA records, the gold
  answer and Token-F1 for quick sanity.

Designed for the final video demo: left sidebar picks model + condition,
main area shows the answer and retrieved evidence side-by-side, and an
expander reveals the full chunk text for any retrieved result.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.answer import (
    MODEL_REGISTRY,
    answer_question,
    build_client,
    load_answer_template,
)
from src.cache import Cache
from src.eval import f1_token
from src.retrieval import (
    OpenAIEmbeddingClient,
    load_index,
    load_kg_index,
    retrieve,
    retrieve_kg2rag,
    retrieve_temporag_kg,
    retrieve_with_year_filter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "samples" / "10k_chunks.jsonl"
INDEX_DIR = REPO_ROOT / "data" / "embeddings" / "chunks"
TRIPLES_PATH = REPO_ROOT / "data" / "kg" / "filtered" / "triples.jsonl"
ANSWER_PROMPT = REPO_ROOT / "prompts" / "answer_v1.txt"
QA_PATHS = [
    REPO_ROOT / "data" / "qa" / "multihop_filtered.jsonl",
    REPO_ROOT / "data" / "qa" / "home_grown.jsonl",
]

CONDITION_LABELS = {
    "L0": "L0 Vanilla (cosine top-k)",
    "L1": "L1 TimeFilter (cosine + year mask)",
    "L2": "L2 KG²RAG (cosine seeds + 1-hop entity expansion)",
    "L3": "L3 TempoRAG-KG (L2 + triple-level temporal filter)",
}

CORPUS_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


# ──────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading chunk index ...")
def _chunk_index():
    return load_index(INDEX_DIR, CHUNKS_PATH)


@st.cache_resource(show_spinner="Loading KG ...")
def _kg_index():
    return load_kg_index(TRIPLES_PATH)


@st.cache_resource
def _embed_client():
    return OpenAIEmbeddingClient()


@st.cache_resource
def _answer_clients():
    return {cfg: build_client(cfg) for cfg in MODEL_REGISTRY}


@st.cache_resource
def _query_cache():
    return Cache(REPO_ROOT / "data" / "cache" / "query_embed")


@st.cache_resource
def _answer_cache():
    return Cache(REPO_ROOT / "data" / "cache" / "answer")


@st.cache_data
def _answer_template():
    return load_answer_template(ANSWER_PROMPT)


@st.cache_data
def _qa_lookup() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in QA_PATHS:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r["question"].strip().lower()] = r
    return out


# ──────────────────────────────────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TempoRAG-KG demo",
    page_icon="📑",
    layout="wide",
)

load_dotenv()

st.title("TempoRAG-KG — 10-K multi-hop QA demo")
st.caption(
    "Temporal Knowledge Graph-Augmented RAG over 25 SEC 10-K filings "
    "(5 yr × 5 tech mega-caps + 5 complementary tickers)."
)

index = _chunk_index()
kg = _kg_index()
template = _answer_template()
embed_client = _embed_client()
qcache = _query_cache()
acache = _answer_cache()
qa_lookup = _qa_lookup()
answer_clients = _answer_clients()

# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Retrieval setup")
    model_key = st.selectbox(
        "Answer model",
        options=sorted(MODEL_REGISTRY),
        index=sorted(MODEL_REGISTRY).index("gpt-4o-mini")
        if "gpt-4o-mini" in MODEL_REGISTRY else 0,
    )
    provider, model_id = MODEL_REGISTRY[model_key]
    st.caption(f"`{model_id}` via {provider}")

    condition = st.radio(
        "Retrieval condition",
        options=list(CONDITION_LABELS),
        format_func=lambda c: CONDITION_LABELS[c],
        index=3,  # default L3
    )

    top_k = st.slider("Top-k chunks", 1, 10, value=5)
    seed_k = st.slider("Seed-k (for L2/L3)", 1, top_k, value=min(3, top_k))

    if condition in ("L1", "L3"):
        # `key="years_input"` lets the sample-question buttons below
        # pre-populate the multiselect via st.session_state.
        years = st.multiselect(
            "Query years (L1/L3 temporal filter)",
            options=CORPUS_YEARS,
            default=[2023],
            key="years_input",
        )
    else:
        years = []
        st.caption("_(Temporal filter disabled for this condition.)_")

    st.divider()
    st.header("KG stats")
    st.metric("Chunks with triples", f"{len(kg.chunk_to_triples):,} / {len(index.chunk_ids):,}")
    st.metric("Total entities", f"{len(kg.entity_to_chunks):,}")
    total_triples = sum(len(v) for v in kg.chunk_to_triples.values())
    st.metric("Total triples", f"{total_triples:,}")

    st.divider()
    st.header("Sample questions")
    SAMPLES = [
        ("What was Microsoft's revenue in fiscal 2022?", [2022]),
        (
            "Compare Apple's Services revenue to Microsoft's cloud revenue for fiscal year 2022",
            [2022],
        ),
        (
            "How did Amazon's AWS operating income evolve from 2020 to 2023?",
            [2020, 2021, 2022, 2023],
        ),
        (
            "Which company had higher data-center revenue in FY2024, NVIDIA or Intel?",
            [2024],
        ),
        (
            "What was Cisco's total revenue for the fiscal year ending July 27, 2024?",
            [2024],
        ),
    ]
    for i, (sq, sy) in enumerate(SAMPLES):
        if st.button(sq, key=f"sample_{i}"):
            st.session_state["question"] = sq
            if condition in ("L1", "L3"):
                st.session_state["years_input"] = sy
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────

if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_area(
    "Question",
    value=st.session_state["question"],
    height=80,
    placeholder="Ask about revenue, segments, risks, or compare across companies / years ...",
)

col_ask, col_space = st.columns([1, 5])
do_ask = col_ask.button("Ask", type="primary", disabled=not question.strip())

if do_ask:
    q = question.strip()
    st.session_state["question"] = q

    with st.spinner(f"Retrieving via {condition} ..."):
        if condition == "L0":
            chunks = retrieve(q, index, embed_client, qcache, k=top_k)
        elif condition == "L1":
            chunks = retrieve_with_year_filter(
                q, years, index, embed_client, qcache, k=top_k
            )
        elif condition == "L2":
            chunks = retrieve_kg2rag(
                q, index, kg, embed_client, qcache, k=top_k, seed_k=seed_k
            )
        else:  # L3
            chunks = retrieve_temporag_kg(
                q, years, index, kg, embed_client, qcache,
                k=top_k, seed_k=seed_k,
            )

    # Mark which chunks would have been in the vanilla top-k (seeds) so users
    # can see what the KG expansion added.
    seed_cids = {
        r["chunk_id"]
        for r in retrieve(q, index, embed_client, qcache, k=seed_k)
    } if condition in ("L2", "L3") else None

    with st.spinner(f"Asking {model_key} ..."):
        try:
            ans = answer_question(
                q, chunks, answer_clients[model_key], acache,
                template=template, model=model_id,
            )
        except Exception as exc:
            st.error(f"Answer call failed: {type(exc).__name__}: {exc}")
            st.stop()

    gold_rec = qa_lookup.get(q.lower())

    # Layout: left column for answer, right column for retrieved chunks.
    a_col, r_col = st.columns([4, 5])

    with a_col:
        st.subheader("Answer")
        st.info(ans["answer"])
        if ans.get("parse_error"):
            st.warning(f"Parse note: {ans['parse_error']}")
        cache_tag = "✓ cache" if ans.get("cache_hit") else "live call"
        st.caption(f"Model `{model_id}` · {cache_tag}")

        if gold_rec:
            st.markdown("**Gold answer** _(matched from labeled QA)_")
            gold = gold_rec["answer"]
            st.success(gold if isinstance(gold, str) else ", ".join(gold))
            f1 = f1_token(ans["answer"], gold)
            st.metric("Token-F1 vs gold", f"{f1:.3f}")
            st.caption(
                f"scope={gold_rec.get('scope')}  ·  hop={gold_rec.get('hop_count')}"
                f"  ·  source={gold_rec.get('source_dataset')}"
            )

    with r_col:
        st.subheader(f"Retrieved chunks ({len(chunks)})")
        for i, c in enumerate(chunks, 1):
            kg_flag = ""
            if seed_cids is not None and c["chunk_id"] not in seed_cids:
                kg_flag = " ·  KG-expanded"
            with st.expander(
                f"[{i}] {c['chunk_id']} — {c.get('ticker','?')} FY{c.get('fy','?')} "
                f"item {c.get('item','?')}  ·  score {c['retrieval_score']:.3f}{kg_flag}",
                expanded=(i <= 2),
            ):
                st.write(c.get("text", "(missing)"))
                n_tr = len(kg.chunk_to_triples.get(c["chunk_id"], []))
                if n_tr:
                    st.caption(f"{n_tr} triples extracted from this chunk")

else:
    st.info(
        "Pick a condition on the left, type a question above (or click a "
        "sample), then press **Ask**. The app will show retrieved chunks "
        "and the model's answer side-by-side. Known questions (from the "
        "129 labeled set) also show gold + F1."
    )
