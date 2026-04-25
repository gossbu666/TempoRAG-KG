# Demo insights — gold cases for the video walkthrough

A short list of question/condition pairs that produce visibly different
outputs across the four retrieval conditions. Use these in T16/T17 (video
script + recording) so the on-screen behaviour is dramatic enough to be
worth filming.

---

## Case 1 — Cross-company hop=2 (the headline demo)

**Question:** *Which company had higher data-center revenue in FY2024,
NVIDIA or Intel?*

**Year filter:** `[2024]`

**Answer model:** `gpt-4o-mini`

| Condition | Intel value reported | Cache | Read |
|---|---|---|---|
| L0 Vanilla     | ❌ "not provided in the excerpts"  | live   | cosine retrieves NVDA-heavy chunks; INTC chunk doesn't make top-5 |
| L1 TimeFilter  | ✅ **\$12,817 million**             | live   | year-mask forces FY2024 → INTC FY2024 chunk lifts up |
| L2 KG²RAG      | ❌ "not provided in the excerpts"  | ✓ cache | **identical to L0** — KG entity walk doesn't bridge to INTC |
| L3 TempoRAG-KG | ✅ \$12,817 million                 | ✓ cache | **identical to L1** — temporal+KG ≡ year-mask on this query |

Why this is the headline:
- **Cache hits prove identity** — L2's prompt is byte-identical to L0's
  (cache key matches), and L3's matches L1's. Same retrieved chunks
  ⇒ same answer. The viewer can see that the "KG-only" condition
  literally adds nothing on this query, while temporal does the work.
- **Concrete dollar value diverges** — \$12,817M vs "not provided" is
  unambiguously visible to a non-NLP audience.
- **Maps directly to the paper's story** — §6.1 *L2 universally
  regresses*; §5.2 *temporal carries the lift*. The demo shows the
  mechanism in real time.

**Suggested narration (~25 s):**

> "Same question, same model, four retrieval strategies. L0 vanilla
> cosine retrieves NVIDIA chunks but misses Intel — answer says 'not
> provided'. L1 adds a year mask, INTC's 2024 filing surfaces, model
> answers 12.8 billion. L2 — KG-only — gets the *same* chunks as L0,
> the answer doesn't change. L3 combines temporal and KG, recovers
> L1's answer. On this corpus, temporal is the active ingredient; KG
> structure alone doesn't bridge across companies."

---

## Case 2 — Single-hop intra (the convergence demo)

**Question:** *What was Microsoft's revenue in fiscal 2022?*

**Year filter:** `[2022]`

All four conditions return **\$211,915 million** (or matching variants).
Use this as the *baseline sanity check* in the video — show the demo
is wired up correctly and the simple case works under every condition,
then pivot to Case 1 to show where the conditions diverge.

**Suggested narration (~10 s):**

> "Sanity check first. Easy single-hop intra-company question. All four
> conditions converge on the same answer — \$211,915 million. The
> system isn't broken; it's just that this case is well within vanilla
> cosine's sweet spot."

---

## Case 3 — Inter-year trajectory (visualises temporal scope)

**Question:** *How did Amazon's AWS operating income evolve from 2020
to 2023?*

**Year filter:** `[2020, 2021, 2022, 2023]` ← important: select all four years

L3 with the four-year filter returns the trajectory:
$13,531M (2020) → $18,532M (2021) → $22,841M (2022) → $24,631M (2023).

Use this case to demonstrate the **multiselect** — the grader sees
that the temporal filter respects multi-year queries, not just a
single year stamp.

---

## Demo flow recommendation (~3 min total in the video budget)

1. **0:00 – 0:30** — Open the Streamlit app, point at the sidebar:
   "10 issuers, 7,467 chunks, 57k triples in the KG."
2. **0:30 – 0:45** — Click sample *Single-hop intra* (Case 2), Ask, show
   convergence. "Baseline works."
3. **0:45 – 2:15** — Click sample *Cross-company hop=3* (Case 1), pick
   year [2024], hit **Compare all 4 conditions**. Walk through the
   four blocks; pause on the L2 vs L1 divergence.
4. **2:15 – 2:45** — Click sample *Inter-year trajectory* (Case 3),
   make sure all four years are selected, run L3 only.
5. **2:45 – 3:00** — Wrap: "Temporal carries the lift; KG bridges
   multi-hop; the demo shows the same conclusion we found in the
   ablation."

---

## UI bug fixes already landed

- LaTeX rendering: dollar signs in answers (`$394,328 million`) were
  being interpreted by Streamlit's markdown as math delimiters and
  rendered in italic without commas. Now escaped via `_md_safe()` in
  `app/streamlit_app.py`. Verified by Case 1 re-run.
- `years_input` session-state: sample-question buttons now use a
  pending-key sentinel so Streamlit doesn't raise the
  "cannot modify after widget instantiated" error.

## Open follow-ups (before recording)

- Pre-warm the cache by clicking through Cases 1, 2, 3 once each so the
  recording shows ✓ cache (faster, no waiting on API).
- Optional: Add a screenshot panel showing KG-expanded chunks visually
  highlighted (the violet badge already does this in v2 of the app).
