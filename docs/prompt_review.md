# Prompt Review — `prompts/extract_v1.txt`

**Status:** Awaiting teammate review. Required by T5 acceptance criteria
(`tasks/plan.md` §5 T5). Gate for Checkpoint 1 → Phase 2.

---

## Review checklist

A reviewer (non-author) should confirm each of the following before sign-off:

- [ ] Output schema is unambiguous and easy to validate programmatically.
- [ ] Multi-tenure rule is explicit ("one triple per continuous period") and
      the example demonstrates two separate triples rather than a merged span.
- [ ] Relative-reference rule forces `valid_from = valid_to = null` and sets
      `metadata.temporal_type = "relative"`. No silent year guessing.
- [ ] `evidence` field is defined as a verbatim substring of the passage.
- [ ] Three worked examples cover: explicit year, multi-tenure, relative.
- [ ] Output contains no markdown / code fences — just JSON.
- [ ] Confidence scale is defined (0.0 – 1.0).
- [ ] No ambiguity between `metadata.temporal_type` values
      (`explicit` / `relative` / `unknown`).

## Reviewer sign-off

| Reviewer | Date | Decision | Notes |
|---|---|---|---|
| _pending_ | | | |

## Notes / follow-ups

(Record any prompt changes requested during review here. If the prompt is
revised, bump the filename to `extract_v2.txt` and create a new review entry.)
