# Notebooks layer

Thin notebooks that sit on top of `src/`. They exist for **inspection,
comparison, and report-ready tables/plots** — nothing else.

## The hard rule

**No business logic in notebook cells.** A cell may:

- import from `src.*`
- call functions from `src.*`
- load files written by `src.*`
- display data (`pandas.DataFrame`, `matplotlib`, `print`)

A cell may **not**:

- define classes or non-trivial functions (move them to `src/`)
- make paid LLM calls without going through `src.cache.Cache`
- bypass cost guards defined in `src/`
- mutate files committed to the repo (other than the notebook itself)

Why: this preserves what Phase 1 bought us — testable logic, deterministic
sampling, cache-routed cost — while still giving us run-and-see-output
ergonomics for the Progress Report.

## File map

| Notebook | Runnable when | Depends on |
|---|---|---|
| `00_samples.ipynb` | **today** | T1 outputs in `data/samples/*.json` |
| `01_pilot.ipynb` | after T6 | `src/kg_extract.py`, `results/pilot/*` |
| `02_kg_overview.ipynb` | after T8 | `results/kg/graph.pkl`, `results/kg/build_stats.json` |
| `03_baseline_vs_full.ipynb` | after T9 + T15 | `results/runs/kg2rag_baseline.json`, `results/runs/rq2_ablation.json` |
| `04_rq_results.ipynb` | after T13 + T14 + T16 | `results/runs/rq{1,3,4}_*.json` |

Stubs for not-yet-runnable notebooks contain the header markdown and a
`[TODO: T#]` markdown block. When the underlying task lands, replace the
TODO block with code cells that import from `src/`.

## Commit policy: keep outputs

Notebooks are committed **with their outputs**. Outputs (tables, plots, sample
text) are part of the deliverable for the Progress Report and must be visible
on GitHub without anyone re-running the cell.

Cost: noisy diffs on output cells. When reviewing a notebook PR, compare
code-only with:

```bash
jupyter nbconvert --to script --stdout notebooks/<name>.ipynb
```

If output diffs become a recurring problem, we can add `nbstripout` as a
pre-commit hook later. Don't pre-optimize.

## Running a notebook

From the repo root:

```bash
source venv/bin/activate
jupyter lab notebooks/        # interactive
# or, headless re-execute (CI-friendly):
jupyter nbconvert --to notebook --execute --inplace notebooks/00_samples.ipynb
```

The notebooks find the project root themselves (walk up looking for `.git`),
so the kernel can be launched from `notebooks/` or from the repo root.

## What to import from `src/`

| Need | Use |
|---|---|
| Temporal classifier | `from src.sampling import classify` |
| Re-run sampling | `from src.sampling import sample_hotpot, sample_musique` |
| Token F1 / EM | `from src.eval import f1_token, em` |
| Aggregate with bootstrap CI | `from src.eval import aggregate` |
| Cache-gated LLM call | `from src.cache import Cache` |
| Krippendorff α | `from src.iaa import krippendorff_alpha_interval` |

If you find yourself wanting something that isn't in `src/`, **add it to
`src/` with a test first** — don't put it in the notebook.
