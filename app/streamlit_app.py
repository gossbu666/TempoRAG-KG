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


def _md_safe(s: str) -> str:
    """Escape dollar signs so Streamlit's markdown renderer doesn't
    interpret them as LaTeX math delimiters. Financial answers are full
    of `$394,328 million`-style strings; without escaping, Streamlit
    renders them in italic math mode and drops the commas.
    """
    if not isinstance(s, str):
        return str(s)
    return s.replace("$", "\\$")


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

# Apply any pending session-state writes BEFORE widgets are instantiated.
# Streamlit forbids assigning to session_state[key] after the widget with
# that key has been created in the same render, so sample-question
# buttons stash their target year set under a sentinel key and we copy
# it across here on the next render.
if "_pending_years" in st.session_state:
    st.session_state["years_input"] = st.session_state.pop("_pending_years")

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
        ("Single-hop intra (easy)",
         "What was Microsoft's revenue in fiscal 2022?",
         [2022]),
        ("Cross-company (KG should help)",
         "Compare Apple's Services revenue to Microsoft's cloud revenue "
         "for fiscal year 2022",
         [2022]),
        ("Inter-year trajectory",
         "How did Amazon's AWS operating income evolve from 2020 to 2023?",
         [2020, 2021, 2022, 2023]),
        ("Inter-year AWS (headline demo)",
         "How did Amazon's AWS segment net sales change from fiscal "
         "2019 to fiscal 2021?",
         [2019, 2021]),
        ("Fiscal-vs-calendar boundary",
         "What was Cisco's total revenue for the fiscal year ending "
         "July 27, 2024?",
         [2024]),
    ]
    for i, (label, sq, sy) in enumerate(SAMPLES):
        if st.button(label, key=f"sample_{i}", help=sq, use_container_width=True):
            st.session_state["question"] = sq
            # Stash the target years under a sentinel; the next render
            # applies it to `years_input` before the multiselect runs.
            if condition in ("L1", "L3"):
                st.session_state["_pending_years"] = sy
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
    placeholder="e.g. Compare Apple's Services revenue to Microsoft's cloud "
                "revenue for fiscal year 2022",
    label_visibility="collapsed",
)

col_ask, col_compare, _ = st.columns([1, 2, 5])
do_ask = col_ask.button("Ask", type="primary", disabled=not question.strip())
do_compare = col_compare.button(
    "Compare all 4 conditions",
    disabled=not question.strip(),
    help="Run L0 / L1 / L2 / L3 on the same question and stack the four answers.",
)


def _retrieve_for(cond: str, q: str) -> list[dict]:
    """Dispatch to the right retriever for a given condition."""
    if cond == "L0":
        return retrieve(q, index, embed_client, qcache, k=top_k)
    if cond == "L1":
        return retrieve_with_year_filter(q, years, index, embed_client, qcache, k=top_k)
    if cond == "L2":
        return retrieve_kg2rag(q, index, kg, embed_client, qcache, k=top_k, seed_k=seed_k)
    return retrieve_temporag_kg(q, years, index, kg, embed_client, qcache,
                                k=top_k, seed_k=seed_k)


def _kg_seed_set(q: str) -> set[str]:
    """Vanilla top-seed_k chunk_ids — used to flag KG-expanded chunks."""
    return {r["chunk_id"] for r in retrieve(q, index, embed_client, qcache, k=seed_k)}


def _render_chunk(i: int, c: dict, seed_cids: set[str] | None,
                  *, expanded: bool) -> None:
    is_kg_expanded = seed_cids is not None and c["chunk_id"] not in seed_cids
    badge = "  :violet-background[🧬 KG-expanded]" if is_kg_expanded else ""
    label = (
        f"[{i}] `{c['chunk_id']}` — {c.get('ticker','?')} FY{c.get('fy','?')} "
        f"item {c.get('item','?')}  ·  score {c['retrieval_score']:.3f}{badge}"
    )
    with st.expander(label, expanded=expanded):
        st.write(c.get("text", "(missing)"))
        n_tr = len(kg.chunk_to_triples.get(c["chunk_id"], []))
        if n_tr:
            st.caption(f"{n_tr} triples extracted from this chunk")


def _ask_one(cond: str, q: str) -> tuple[list[dict], dict, set[str] | None]:
    """Returns (chunks, answer, seed_cids)."""
    chunks = _retrieve_for(cond, q)
    seed_cids = _kg_seed_set(q) if cond in ("L2", "L3") else None
    ans = answer_question(
        q, chunks, answer_clients[model_key], acache,
        template=template, model=model_id,
    )
    return chunks, ans, seed_cids


if do_ask or do_compare:
    q = question.strip()
    st.session_state["question"] = q
    gold_rec = qa_lookup.get(q.lower())

    if do_compare:
        st.markdown("### Comparison across all 4 retrieval conditions")
        st.caption(
            "Same question, same model, four retrieval strategies. "
            "Watch for differences in retrieved chunks (right panel) and "
            "in the answer (left panel)."
        )
        if gold_rec:
            gold = gold_rec["answer"]
            gold_str = gold if isinstance(gold, str) else ", ".join(gold)
            st.success(f"**Gold answer:** {_md_safe(gold_str)}")
            st.caption(
                f"scope={gold_rec.get('scope')} · hop={gold_rec.get('hop_count')}"
                f" · source={gold_rec.get('source_dataset')}"
            )

        for cond in ("L0", "L1", "L2", "L3"):
            with st.spinner(f"Running {CONDITION_LABELS[cond]} ..."):
                chunks, ans, seed_cids = _ask_one(cond, q)
            f1_str = ""
            if gold_rec:
                gold = gold_rec["answer"]
                f1 = f1_token(ans["answer"], gold)
                f1_str = f"  ·  F1 = **{f1:.3f}**"

            st.markdown(f"#### {CONDITION_LABELS[cond]}{f1_str}")
            a_col, r_col = st.columns([4, 5])
            with a_col:
                st.info(_md_safe(ans["answer"]))
                cache_tag = "✓ cache" if ans.get("cache_hit") else "live call"
                st.caption(f"`{model_id}` · {cache_tag}")
            with r_col:
                with st.expander(f"Retrieved chunks ({len(chunks)})", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        _render_chunk(i, c, seed_cids, expanded=False)
            st.divider()

    else:  # single-condition Ask
        with st.spinner(f"Retrieving via {CONDITION_LABELS[condition]} ..."):
            chunks = _retrieve_for(condition, q)
        seed_cids = _kg_seed_set(q) if condition in ("L2", "L3") else None

        with st.spinner(f"Asking {model_key} ..."):
            try:
                ans = answer_question(
                    q, chunks, answer_clients[model_key], acache,
                    template=template, model=model_id,
                )
            except Exception as exc:
                st.error(f"Answer call failed: {type(exc).__name__}: {exc}")
                st.stop()

        a_col, r_col = st.columns([4, 5])
        with a_col:
            st.subheader("Answer")
            st.info(_md_safe(ans["answer"]))
            if ans.get("parse_error"):
                st.warning(f"Parse note: {ans['parse_error']}")
            cache_tag = "✓ cache" if ans.get("cache_hit") else "live call"
            st.caption(f"Model `{model_id}` · {cache_tag}")

            if gold_rec:
                st.markdown("**Gold answer** _(matched from labeled QA)_")
                gold = gold_rec["answer"]
                gold_str = gold if isinstance(gold, str) else ", ".join(gold)
                st.success(_md_safe(gold_str))
                f1 = f1_token(ans["answer"], gold)
                st.metric(
                    "Token-F1 vs gold",
                    f"{f1:.3f}",
                    help="SQuAD-style word-level overlap between the model's "
                         "answer and the gold. Range 0–1; 1 = identical bag of "
                         "tokens after lowercase + stop-punct + stop-words.",
                )
                st.caption(
                    f"scope={gold_rec.get('scope')}  ·  hop={gold_rec.get('hop_count')}"
                    f"  ·  source={gold_rec.get('source_dataset')}"
                )

        with r_col:
            st.subheader(f"Retrieved chunks ({len(chunks)})")
            for i, c in enumerate(chunks, 1):
                _render_chunk(i, c, seed_cids, expanded=(i <= 2))

else:
    # Demo-friendly empty state with three quick-start steps.
    st.markdown(
        """
        ### Try it out

        **TempoRAG-KG** evaluates four retrieval strategies on the same
        question, all backed by the same 25-filing 10-K corpus and the same
        knowledge graph (~58k triples).

        1. **Type a question** above (or click a sample on the left).
        2. **Pick a retrieval condition** in the sidebar — `L0` is plain
           cosine, `L3` adds a knowledge-graph walk filtered by temporal
           validity.
        3. **Press *Ask*** for one condition, or **Compare all 4** to see
           every condition stacked on the same question.

        When the question matches one of our 129 labelled QA items, the
        gold answer and Token-F1 score are also shown.
        """
    )
