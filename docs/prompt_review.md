# Prompt Review — `prompts/extract_v1.txt` (financial domain, v2)

**Status:** ✍️ Draft awaiting teammate review.
**Author:** Supanut (2026-04-18).
**Reviewer(s):** _unassigned — pick one teammate before P1_.
**Artifact:** [prompts/extract_v1.txt](../prompts/extract_v1.txt) (financial).
**Archive:** [prompts/archive/extract_v1_wikipedia.txt](../prompts/archive/extract_v1_wikipedia.txt) (v1 Wikipedia version, frozen for reference).

Supersedes the earlier Wikipedia-era review checklist that used to live in this file (see git history). After the TA pivot on 2026-04-16 the prompt's domain changed, so the review criteria changed with it.

---

## 1. Why a review gate (before we spend money)

The prompt is the contract between the LLM and the KG. A bug that passes unreviewed here means:

1. We run 3,810 chunks through Gemini Flash at ~$1.07.
2. The triples come out subtly wrong — e.g. fiscal years off by one, or "we expect" statements mixed in with historical facts.
3. We don't notice until the KG is built and RQ4 numbers look weird, by which point the money is spent and the pilot budget is half-gone.

P1 is a 20-chunk pilot designed to catch this before the full run, but the prompt should enter the pilot already sanity-checked by a human who reads 10-K prose for a living — or at least a human who reads it critically. That's the job of this review.

## 2. What changed vs v1 (Wikipedia)

| Area | v1 (Wikipedia) | v2 (financial) |
|---|---|---|
| Domain | Biographies, sports, history | Corporate financials, operations, forward guidance |
| Subject canonicalization | Passage-local ("Barack Obama") | Resolve "we / the Company / our" → `company_name` from chunk metadata |
| Temporal resolution | Passage-local ("from 2009 to 2017") | Passage + **chunk metadata** ("fiscal 2022" → 2022; "prior year" → `fiscal_year − 1`) |
| `temporal_type` values | `explicit` / `relative` / `unknown` | Added **`forward_looking`** |
| Object type | Always string | String OR number (for $211.9B, 41%, etc.) |
| Predicate convention | Short snake_case label | Unit suffix required for numeric objects (`_usd_billion`, `_percent`, `_bps`, `_headcount`) |
| New rule sections | Multi-tenure (Obama/Cleveland) | **Period-over-period**, **forward-looking guidance**, **what-not-to-extract** (boilerplate / hypothetical risk prose) |
| Template placeholders | `{{PASSAGE}}` | `{{CHUNK_METADATA}}` + `{{PASSAGE}}` |

## 3. Review checklist — tick before sign-off

Open [prompts/extract_v1.txt](../prompts/extract_v1.txt) and assess.

### Correctness of the rules

- [ ] **Fiscal-year convention.** I've set `fiscal 2022 → valid_from = valid_to = 2022`, using the calendar year in which the fiscal year ENDS. This matches how 10-Ks self-label. Any objection? (MSFT's fiscal 2022 ends June 2022 — we label it 2022, not 2021. AAPL's fiscal 2022 ends September 2022 — same.)
- [ ] **"Prior year" resolution.** Resolved deterministically to `fiscal_year − 1` as `explicit`, not `relative`. Any case in 10-K prose where "prior year" is genuinely ambiguous in a way that `explicit` would be wrong?
- [ ] **Forward-looking trigger phrases.** Current list: *"we expect", "we anticipate", "we plan to", "we intend to", "next year", "in the coming year", "for fiscal YYYY+1", "going forward"*. Anything missing? Any false-positive traps? (e.g. *"we plan to continue a historical practice that began in 2018"* is retrospective, not forward-looking.)
- [ ] **"What NOT to extract" section.** Does the line between *risk factor as category* ("cyberattack") and *hypothetical outcome* ("the Company could suffer material harm") feel right for Item 1A chunks, or too loose?

### Quality of the 3 worked examples

- [ ] **Example 1 (AAPL fiscal 2022, explicit years).** I emit `net_sales_usd_billion` for BOTH 2022 and 2021 from one sentence. Does that split match what you'd want for the KG?
- [ ] **Example 2 (MSFT forward-looking).** Two triples: one with `valid_to = null` (open-ended expected trend), one with `valid_to = 2024` (dated specific intent). Does that split feel right?
- [ ] **Example 3 (MSFT prior-year comparison).** Four triples from one paragraph — right density, or too aggressive?

### Predicate naming convention

- [ ] Are unit-suffixed predicates (`revenue_usd_billion`, `operating_margin_percent`) acceptable? Alternative: unit in metadata. I picked in-predicate because it keeps triples self-describing and avoids a unit-lookup step during KG query.
- [ ] Normalize across USD-billion vs USD-million? (MSFT uses billions in MD&A; some Item 8 notes use millions.) For now I let the LLM pick the unit matching the passage; downstream we can normalize.

### Schema compliance

- [ ] The `{subject, predicate, object, valid_from, valid_to, confidence, evidence, metadata}` schema is unchanged from v1 except `object` can now be numeric. OK for downstream P1 code? (I believe yes — Python `json.loads` handles both.)

### Open questions for the reviewer

- [ ] Canonicalize "the Intelligent Cloud segment" to "Microsoft — Intelligent Cloud"? (Currently: left raw. KG can merge later.)
- [ ] Any Risk-Factor (Item 1A) patterns the current rules handle badly? That section is the most adversarial for extraction because of the hypothetical voice.

## 4. How to leave review comments

Either:
- Append a ` Review by <name> (<date>)` section at the bottom of this file with ticked checkboxes + free-form notes, OR
- Leave inline comments on the PR when the prompt + review doc are pushed.

No GitHub PR open yet — comments directly on this file are fine for now.

## 5. Gate

The prompt is not considered ready for P1 until:

- [ ] At least one teammate signs off below.
- [ ] Blocker-severity issues identified above are addressed in a prompt revision.
- [ ] Supanut confirms the prompt file is unchanged between review and pilot run (no silent edits post-approval).

---

## Review by _TBD_ (YYYY-MM-DD)

_Paste ticked checklist + free-form notes here._
