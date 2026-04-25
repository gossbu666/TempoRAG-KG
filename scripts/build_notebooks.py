"""Generate the notebooks/*.ipynb scaffolds from a single source of truth.

Re-run after editing this file to regenerate. Notebooks are then opened in
JupyterLab / VSCode and executed; outputs are committed alongside the cells.
See notebooks/README.md for the design contract.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"},
    }
    out = NB_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)}")


SETUP_CELL = '''import json
import os
import sys
from pathlib import Path


def find_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    raise RuntimeError(f"project root not found from {start}")


ROOT = find_root(Path.cwd())
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

print(f"Project root: {ROOT}")
'''


# ---------------- 00 samples (runnable today) ----------------

def nb_00() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 00 — Samples overview\n\n"
            "**Purpose.** Sanity-check the deterministic samples produced by `src.sampling` "
            "(task T1). Show counts, temporal pattern frequencies, and random example questions "
            "for both datasets, then verify the committed sample files re-derive byte-identically.\n\n"
            "**Inputs**: `data/samples/hotpot_1000.json`, `data/samples/musique_500.json`, "
            "raw datasets in `data/`.\n\n"
            "**Outputs**: display only (no files written).\n\n"
            "**Env**: reads `RANDOM_SEED` from `.env` via `src.sampling`."
        ),
        code(SETUP_CELL),
        md("## HotpotQA — temporal vs non-temporal counts"),
        code(
            'HOTPOT = json.loads((ROOT / "data/samples/hotpot_1000.json").read_text())\n'
            'print(f"HotpotQA sample: {len(HOTPOT)} questions")\n'
            '\n'
            'counts = pd.Series([r["temporal"] for r in HOTPOT]).value_counts()\n'
            'counts.index = ["temporal" if x else "non-temporal" for x in counts.index]\n'
            'counts.to_frame("count")\n'
        ),
        md("## HotpotQA — temporal pattern frequency (in temporal subset)"),
        code(
            'from collections import Counter\n'
            '\n'
            'temporal_n = sum(1 for r in HOTPOT if r["temporal"])\n'
            'pat_counts = Counter(p for r in HOTPOT for p in r["patterns"])\n'
            'df = pd.DataFrame(pat_counts.most_common(), columns=["pattern", "count"])\n'
            'df["pct_of_temporal"] = (df["count"] / temporal_n * 100).round(1)\n'
            'df\n'
        ),
        md("## MuSiQue — hop-count distribution"),
        code(
            'MUSIQUE = json.loads((ROOT / "data/samples/musique_500.json").read_text())\n'
            'print(f"MuSiQue sample: {len(MUSIQUE)} questions")\n'
            '\n'
            'hop_counts = pd.Series([r["hop_count"] for r in MUSIQUE]).value_counts().sort_index()\n'
            'hop_counts.to_frame("count")\n'
        ),
        md("## MuSiQue — temporal pattern frequency"),
        code(
            'pat_counts_m = Counter(p for r in MUSIQUE for p in r["patterns"])\n'
            'pd.DataFrame(pat_counts_m.most_common(), columns=["pattern", "count"])\n'
        ),
        md("## Random example questions (seeded)"),
        code(
            'import random\n'
            '\n'
            'rng = random.Random(42)\n'
            '\n'
            'def show(records, label, n=5):\n'
            '    temp = [r for r in records if r["temporal"]]\n'
            '    non_temp = [r for r in records if not r["temporal"]]\n'
            '    print(f"=== {label}: {n} temporal ===")\n'
            '    for r in rng.sample(temp, min(n, len(temp))):\n'
            '        print(f"  [{r[\\"id\\"]}] {r[\\"question\\"]}")\n'
            '        if r["patterns"]:\n'
            '            print(f"    patterns: {\\", \\".join(r[\\"patterns\\"])}")\n'
            '    print(f"\\n=== {label}: {n} non-temporal ===")\n'
            '    for r in rng.sample(non_temp, min(n, len(non_temp))):\n'
            '        print(f"  [{r[\\"id\\"]}] {r[\\"question\\"]}")\n'
            '    print()\n'
            '\n'
            'show(HOTPOT, "HotpotQA", n=5)\n'
            'show(MUSIQUE, "MuSiQue", n=5)\n'
        ),
        md(
            "## SHA-stability sanity check\n\n"
            "Re-runs `src.sampling.sample_hotpot()` and `sample_musique()` against the raw data "
            "and confirms the SHA256 matches the committed sample files. If this drifts, the "
            "deterministic guarantee from T1 is broken."
        ),
        code(
            'import hashlib\n'
            'import tempfile\n'
            '\n'
            'from src.sampling import sample_hotpot, sample_musique\n'
            '\n'
            'EXPECTED = {\n'
            '    "hotpot": "5dcdb24e4152fc956a1809dbb19830b472156c542e127c92bfc83ee3cd0cc598",\n'
            '    "musique": "bff1ac8a57b7fb1565a3231334792a608684d9138a61fc6ac70b99331a26da19",\n'
            '}\n'
            '\n'
            'with tempfile.TemporaryDirectory() as td:\n'
            '    td = Path(td)\n'
            '    sample_hotpot("data/hotpot_dev_distractor_v1.json", td / "h.json")\n'
            '    sample_musique("data/musique_ans_v1.0_dev.jsonl", td / "m.json")\n'
            '    h_h = hashlib.sha256((td / "h.json").read_bytes()).hexdigest()\n'
            '    h_m = hashlib.sha256((td / "m.json").read_bytes()).hexdigest()\n'
            '\n'
            'rows = [\n'
            '    ("hotpot",  h_h, EXPECTED["hotpot"],  h_h == EXPECTED["hotpot"]),\n'
            '    ("musique", h_m, EXPECTED["musique"], h_m == EXPECTED["musique"]),\n'
            ']\n'
            'pd.DataFrame(rows, columns=["dataset", "re_sample_sha256", "committed_sha256", "match"])\n'
        ),
    ]


# ---------------- 01 pilot (stub for T6) ----------------

def nb_01() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 01 — Pilot extraction (T6)\n\n"
            "**Status:** stub. Fill in when T6 lands `src/kg_extract.py` and "
            "`scripts/run_pilot.py`. Do **not** make paid Gemini calls outside the "
            "`Cache` wrapper.\n\n"
            "**Cost guard.** Hard abort at $0.50 total (per `tasks/plan.md` §5 T6). "
            "Implemented in `scripts/run_pilot.py`; this notebook only renders results.\n\n"
            "**Inputs (planned)**: 20 chunks sourced from `data/samples/hotpot_1000.json`, "
            "results in `results/pilot/log.jsonl` and `results/pilot/raw.jsonl`.\n\n"
            "**Outputs (planned)**: per-chunk latency / cost / triple counts table, "
            "5 GOOD + 5 PROBLEMATIC examples, projected full-run cost, GO/NO-GO recommendation."
        ),
        code(SETUP_CELL),
        md(
            "## TODO — fill during T6\n\n"
            "Cells to add:\n\n"
            "1. Load 20 chunks from sampled HotpotQA.\n"
            "2. Run `extract_triples()` per chunk via `Cache` (already cached if pilot has run).\n"
            "3. Display per-chunk dataframe: `latency_s`, `est_cost_usd`, `parsed_triples_n`, `parse_failures`.\n"
            "4. Show 5 GOOD + 5 PROBLEMATIC triple examples with raw response.\n"
            "5. Aggregate: avg triples/chunk, non-null validity rate, projected full-run cost vs $5 cap.\n"
            "6. **GO / NO-GO recommendation cell** — fill manually before committing the executed notebook."
        ),
    ]


# ---------------- 02 KG overview (stub for T8) ----------------

def nb_02() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 02 — Knowledge graph overview (T8)\n\n"
            "**Status:** stub. Fill in when T8 lands `results/kg/graph.pkl` and "
            "`results/kg/build_stats.json`.\n\n"
            "**Checkpoint 2 gate.** Non-null validity coverage rate must be ≥ 30% — if not, "
            "the temporal filter is mostly a no-op and we flag this as a study limitation."
        ),
        code(SETUP_CELL),
        md(
            "## TODO — fill during T8\n\n"
            "Cells to add:\n\n"
            "1. Load `results/kg/graph.pkl` (NetworkX `MultiDiGraph`).\n"
            "2. Display `n_nodes`, `n_edges`, `avg_degree`, `non_null_validity_rate`.\n"
            "3. Plot degree distribution (histogram).\n"
            "4. Sample 10 random edges; show `subject / predicate / object / valid_from / valid_to / source_chunk_id`.\n"
            "5. Load `results/kg/build_stats.json`; show total cost, wall-clock, triples extracted."
        ),
    ]


# ---------------- 03 baseline vs full (stub for T9 + T15) ----------------

def nb_03() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 03 — KG²RAG baseline vs Full TempoRAG-KG (T9 + T15)\n\n"
            "**Status:** stub. Fill in when T9 lands `results/runs/kg2rag_baseline.json` "
            "and T15 lands `results/runs/rq2_ablation.json`.\n\n"
            "**Reproduction target (T9).** F1 within 2.0 points of the KG²RAG paper "
            "(Zhu et al. 2025, Table 3 ≈ 85.6).\n\n"
            "**Cost guard (T15).** Hard $2 cap on the ablation."
        ),
        code(SETUP_CELL),
        md(
            "## TODO — fill during T9 + T15\n\n"
            "Cells to add:\n\n"
            "1. Load `results/runs/kg2rag_baseline.json`; show F1 / EM with bootstrap CI; print delta vs paper.\n"
            "2. Load `results/runs/rq2_ablation.json`; show 3-condition × 2-dataset table on overall / temporal / non-temporal subsets.\n"
            "3. Plot ΔF1 (Full − KG²RAG) on the temporal subset for both datasets.\n"
            "4. Print headline numbers ready to paste into the Progress Report §6."
        ),
    ]


# ---------------- 04 RQ results (stub for T13 + T14 + T16) ----------------

def nb_04() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 04 — Final RQ results for the Progress Report (T13 + T14 + T16)\n\n"
            "**Status:** stub. Fill in as T13 / T14 / T16 land their `results/runs/rq*_*.json` files.\n\n"
            "Tables generated here are intended to paste directly into "
            "`proposal/progress_report.tex` (T17)."
        ),
        code(SETUP_CELL),
        md(
            "## TODO — fill during T13 + T14 + T16\n\n"
            "Cells to add:\n\n"
            "1. **RQ3 extraction** (T13): precision / recall / F1 per pattern type "
            "(explicit year / Type 3b relative / conflicting), with ±1 yr tolerance.\n"
            "2. **RQ1 failure modes** (T14): distribution across the 5-type taxonomy "
            "(Stale / Conflicting / Missing / Type 3b / Hop) with one example question per type.\n"
            "3. **RQ4 generator ablation** (T16): ΔF1 = Full − KG²RAG per generator "
            "(LLaMA-8B / LLaMA-70B / Gemini Flash / GPT-4o-mini); test inverse-scaling hypothesis.\n"
            "4. Combined report-ready table for Progress Report §6 (Updated Methodology + Results)."
        ),
    ]


def main() -> None:
    write("00_samples.ipynb", nb_00())
    write("01_pilot.ipynb", nb_01())
    write("02_kg_overview.ipynb", nb_02())
    write("03_baseline_vs_full.ipynb", nb_03())
    write("04_rq_results.ipynb", nb_04())


if __name__ == "__main__":
    main()
