"""Run `prompts/extract_v1.txt` against a 10-K chunk and return KG triples.

Pipeline:

    chunk record (from `src.sampling.sample_10k_chunks`)
        │
        ▼
    render_prompt()   — substitute {{CHUNK_METADATA}} + {{PASSAGE}}
        │
        ▼
    cache.get(key)    — key = sha256(model | rendered_prompt | params)
        │  hit ────► parse_response() ────► ExtractionResult(cache_hit=True)
        │  miss
        ▼
    client.generate() — single LLM call (Gemini 1.5 Flash by default)
        │
        ▼
    cache.put(key)    — store raw response; never store parsed triples
        │
        ▼
    parse_response()  — strip fences, json.loads, per-triple schema check
        │
        ▼
    ExtractionResult(cache_hit=False)

Design choices:

- **Raw responses are cached, not parsed triples.** If we ever tighten the
  schema validator or fix a parser bug, we can re-parse cached responses
  for free. If we cached the parsed triples, every validator tweak would
  invalidate the cache.

- **Parse errors do not raise.** A bad JSON response from the LLM is a
  data point, not a crash. `ExtractionResult.parse_errors` carries the
  reasons; downstream code (the pilot `run_pilot.py`) decides whether to
  retry, discard, or flag the chunk.

- **Per-triple validation, not all-or-nothing.** One malformed triple in
  a batch of 26 no longer forces the whole chunk to be discarded. Valid
  triples pass through; invalid ones are dropped with per-triple error
  strings. A fatal error (malformed JSON, missing `triples` key) yields
  `triples == []` and a single error string — that remains all-or-nothing
  because there is nothing recoverable.

- **`client` is an abstract protocol.** Tests inject a FakeClient that
  returns canned responses — we never hit the network during pytest.
  `GeminiClient` is a concrete adapter at the bottom of this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.cache import Cache


# -----------------------------------------------------------------------------
# Canonical company names per ticker. Lives with the extractor because the
# prompt's "resolve first-person references" rule consumes it. Updating the
# corpus (new tickers) means updating this dict.
# -----------------------------------------------------------------------------
DEFAULT_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "ORCL": "Oracle Corporation",
    "INTC": "Intel Corporation",
    "NVDA": "NVIDIA Corporation",
    "ADBE": "Adobe Inc.",
}

# Values allowed in triple.metadata.temporal_type. Must stay in sync with the
# prompt — see prompts/extract_v1.txt §"TEMPORAL TYPE DECISION TREE".
ALLOWED_TEMPORAL_TYPES: frozenset[str] = frozenset(
    {"explicit", "relative", "forward_looking", "unknown"}
)

REQUIRED_TRIPLE_KEYS: frozenset[str] = frozenset(
    {"subject", "predicate", "object", "valid_from", "valid_to",
     "confidence", "evidence", "metadata"}
)


# -----------------------------------------------------------------------------
# Client protocol + structured result
# -----------------------------------------------------------------------------

class LLMClient(Protocol):
    """Anything with a `.generate(prompt, temperature=...) -> str` works here.

    The return value is the raw text of the model response. Parsing +
    validation happen in this module, not in the client.
    """

    def generate(self, prompt: str, temperature: float = 0.0) -> str: ...


@dataclass
class ExtractionResult:
    chunk_id: str
    model: str
    cache_hit: bool
    raw_response: str
    triples: list[dict]
    parse_errors: list[str] = field(default_factory=list)
    # Fields populated by the caller (run_pilot.py) if it wants to track cost.
    # Left as a free-form dict so we don't couple this module to a specific
    # token-accounting library.
    usage: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Prompt rendering
# -----------------------------------------------------------------------------

def format_chunk_metadata(
    chunk: dict,
    company_name_map: dict[str, str] = DEFAULT_COMPANY_NAMES,
) -> str:
    """Render the CHUNK METADATA block exactly as the prompt's examples show.

    Fields consumed from the chunk record (produced by sample_10k_chunks):
      ticker, fy, item, filing_date, period_of_report
    """
    ticker = chunk["ticker"]
    company = company_name_map.get(ticker, ticker)  # fall back to ticker if unknown
    return (
        f"    ticker:           {ticker}\n"
        f"    company_name:     {company}\n"
        f"    filing_date:      {chunk['filing_date']}\n"
        f"    period_of_report: {chunk['period_of_report']}\n"
        f"    fiscal_year:      {chunk['fy']}\n"
        f"    item:             {chunk['item']}"
    )


def render_prompt(
    template: str,
    chunk: dict,
    company_name_map: dict[str, str] = DEFAULT_COMPANY_NAMES,
) -> str:
    """Substitute both template placeholders and return the final prompt string."""
    if "{{CHUNK_METADATA}}" not in template or "{{PASSAGE}}" not in template:
        raise ValueError(
            "prompt template must contain both {{CHUNK_METADATA}} and {{PASSAGE}}"
        )
    metadata_block = format_chunk_metadata(chunk, company_name_map)
    return (
        template
        .replace("{{CHUNK_METADATA}}", metadata_block)
        .replace("{{PASSAGE}}", chunk["text"])
    )


def load_prompt_template(path: Path | str = "prompts/extract_v1.txt") -> str:
    """Convenience loader — read the prompt file from disk."""
    return Path(path).read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Response parsing + validation
# -----------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _strip_code_fence(text: str) -> str:
    """The prompt says 'no markdown fences', but models sometimes add them anyway.
    Strip a leading ```json ... ``` wrapper if present; otherwise return as-is.
    """
    stripped = text.strip()
    m = _CODE_FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def _strip_ws(s: str) -> str:
    """Remove ALL whitespace.

    10-K PDF extraction scatters whitespace through tables (each cell on
    its own line, `$` separated from the number, `-1 %`, etc.). Models
    read the table visually and emit evidence as humans write it
    ("$25,821", "-1%"), so a substring check that collapses whitespace
    to a single space still fails — the cell gap survives.

    Stripping whitespace entirely before comparison tolerates arbitrary
    PDF-column splits without permitting real content to be fabricated.
    False-positive risk (e.g., "his" matching "this") is the same as the
    existing substring match; it is not made worse by whitespace removal.
    """
    return _WS_RE.sub("", s)


def _numeric_candidates(n: int | float) -> list[str]:
    """String forms a numeric `object` may take in the chunk.

    Financial text writes the same number many ways: `9930`, `9,930`,
    `$9,930`, `9930.0`. Percentages may be stored as whole numbers
    (`25`) or fractions (`0.25`) while the chunk prints `25%`. Negative
    numbers are frequently shown in accounting parentheses: `(1,012)`.
    """
    cands: set[str] = set()
    if isinstance(n, float):
        cands.add(f"{n:g}")                                # "0.64", "9930", "211.9"
        if n == int(n):                                    # whole-number float
            n = int(n)
    if isinstance(n, int):
        cands.add(str(n))                                  # "9930", "-1012"
        cands.add(f"{n:,}")                                # "9,930", "-1,012"
    if isinstance(n, float) and 0 < n < 1:                 # fraction → percentage
        pct = n * 100
        cands |= {f"{pct:g}", f"{pct:g}%"}
    if (isinstance(n, int) and n < 0) or (isinstance(n, float) and n < 0):
        absn = abs(n)
        if isinstance(absn, float) and absn == int(absn):
            absn = int(absn)
        if isinstance(absn, int):
            cands |= {f"({absn})", f"({absn:,})"}          # "(1012)", "(1,012)"
        else:
            cands.add(f"({absn:g})")
    return sorted(cands)


def _verifiable_numeric(obj: Any) -> bool:
    """True when `obj` is a numeric value we can verify against chunk text.
    Booleans are excluded: they are technically ints in Python but have
    no literal textual form to match against the chunk.
    """
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)


def _validate_triple(triple: Any, chunk_text: str) -> str | None:
    """Return an error string if the triple is malformed, else None.

    Hallucination guard — three-branch dispatch on object type:
      • Numeric object (int/float, excl. bool): verify the VALUE appears
        in chunk. Authoritative — no evidence fallback because a
        fabricated number is the most dangerous failure mode, and
        numeric candidates already tolerate PDF-table artifacts
        (`$\\n\\n25,821`, `(1,012)`, `64%` ↔ `0.64`).
      • String object: try matching the object substring against the
        chunk first (catches proper nouns, short spans copied from the
        text). If not found, fall back to the evidence string — this
        admits legitimate paraphrases (e.g., `object="adverse effect
        on advertising business"` for prose that says `"adversely
        affect... our advertising business"`).
      • Bool / None object: only the evidence handle is verifiable.
    """
    if not isinstance(triple, dict):
        return f"not a JSON object (got {type(triple).__name__})"
    missing = REQUIRED_TRIPLE_KEYS - triple.keys()
    if missing:
        return f"missing keys: {sorted(missing)}"
    for year_key in ("valid_from", "valid_to"):
        v = triple[year_key]
        # bool is a subclass of int in Python — reject it explicitly.
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            return f"{year_key} must be integer or null, got {type(v).__name__}"
    conf = triple["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return f"confidence must be number, got {type(conf).__name__}"
    if not 0.0 <= float(conf) <= 1.0:
        return f"confidence out of range [0,1]: {conf}"
    metadata = triple["metadata"]
    if not isinstance(metadata, dict):
        return f"metadata must be object, got {type(metadata).__name__}"
    tt = metadata.get("temporal_type")
    if tt not in ALLOWED_TEMPORAL_TYPES:
        return f"metadata.temporal_type must be one of {sorted(ALLOWED_TEMPORAL_TYPES)}, got {tt!r}"
    evidence = triple["evidence"]
    if not isinstance(evidence, str):
        return f"evidence must be string, got {type(evidence).__name__}"
    obj = triple["object"]
    chunk_stripped = _strip_ws(chunk_text)
    if _verifiable_numeric(obj):
        for cand in _numeric_candidates(obj):
            if _strip_ws(cand) in chunk_stripped:
                return None
        return f"object not found in chunk text: {obj!r}"
    if isinstance(obj, str) and obj and _strip_ws(obj) in chunk_stripped:
        return None
    # Paraphrased str, or bool/None: fall back to the evidence handle.
    if _strip_ws(evidence) not in chunk_stripped:
        preview = evidence[:80] + ("..." if len(evidence) > 80 else "")
        return f"evidence not found in chunk text: {preview!r}"
    return None


def parse_response(raw: str, chunk_text: str) -> tuple[list[dict], list[str]]:
    """Parse the LLM's raw response into (valid_triples, errors).

    Per-triple validation is best-effort: one bad triple in a batch of many
    no longer discards the whole chunk. Every triple that passes schema +
    the hallucination guard flows through; each rejection carries a
    human-readable reason in `errors`.

    Fatal errors (malformed JSON, missing `triples` key, top-level not an
    object) still return `([], [<one error>])` because there is nothing
    partial to salvage from them.

    `chunk_text` is needed for the hallucination guard inside
    `_validate_triple`. Never raises — success iff `errors` is empty.
    """
    body = _strip_code_fence(raw)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return [], [f"JSONDecodeError: {e.msg} at pos {e.pos}"]
    if not isinstance(obj, dict):
        return [], [f"top-level JSON must be an object, got {type(obj).__name__}"]
    if "triples" not in obj:
        return [], ["missing 'triples' key in response"]
    triples = obj["triples"]
    if not isinstance(triples, list):
        return [], [f"'triples' must be an array, got {type(triples).__name__}"]
    valid: list[dict] = []
    errors: list[str] = []
    for i, t in enumerate(triples):
        err = _validate_triple(t, chunk_text)
        if err:
            errors.append(f"triple[{i}]: {err}")
        else:
            valid.append(t)
    return valid, errors


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def extract_triples(
    chunk: dict,
    prompt_template: str,
    client: LLMClient,
    cache: Cache,
    *,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    company_name_map: dict[str, str] = DEFAULT_COMPANY_NAMES,
) -> ExtractionResult:
    """Extract KG triples from a single 10-K chunk.

    Cache semantics:
      - Key = sha256(model | rendered_prompt | {"temperature": t}).
      - The rendered prompt contains the chunk text + metadata, so different
        chunks always produce different keys without needing chunk_id in params.
      - Changing the prompt template busts the cache (correct behavior: the
        contract with the LLM changed).
      - Changing the `temperature` busts the cache.
    """
    rendered = render_prompt(prompt_template, chunk, company_name_map)
    params = {"temperature": temperature}
    key = cache.key_for(model, rendered, params)

    cached = cache.get(key)
    if cached is not None:
        raw = cached["response"]
        triples, errors = parse_response(raw, chunk["text"])
        return ExtractionResult(
            chunk_id=chunk["chunk_id"],
            model=model,
            cache_hit=True,
            raw_response=raw,
            triples=triples,
            parse_errors=errors,
        )

    raw = client.generate(rendered, temperature=temperature)
    cache.put(key, {
        "response": raw,
        "model": model,
        "chunk_id": chunk["chunk_id"],
    })
    triples, errors = parse_response(raw, chunk["text"])
    return ExtractionResult(
        chunk_id=chunk["chunk_id"],
        model=model,
        cache_hit=False,
        raw_response=raw,
        triples=triples,
        parse_errors=errors,
    )


# -----------------------------------------------------------------------------
# Concrete Gemini adapter
# -----------------------------------------------------------------------------
# Kept in this module (not a separate file) so callers can see the whole wire
# format in one place. Tests do NOT import this class — they use FakeClient.
# -----------------------------------------------------------------------------

class GeminiClient:
    """Thin adapter around `google.generativeai` for Gemini 1.5 Flash.

    Import is deferred so the test suite can import this module without
    having the package installed in the environment. The real pilot
    (`scripts/run_pilot.py`) instantiates GeminiClient; tests do not.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str | None = None,
    ) -> None:
        import google.generativeai as genai  # deferred
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set — refuse to run without a key"
            )
        genai.configure(api_key=resolved_key)
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # `response_mime_type: application/json` forces Gemini to emit a JSON
        # body — eliminates one whole class of "model wrapped output in prose".
        resp = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        return resp.text


class OpenAIClient:
    """Thin adapter around the OpenAI Chat Completions API.

    Drop-in replacement for `GeminiClient` — same `.generate(prompt,
    temperature) -> str` shape. Used when the Gemini billing path is
    unavailable or for cost/quality comparison. `response_format=
    {"type": "json_object"}` mirrors Gemini's JSON-only guarantee.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        api_key: str | None = None,
        max_retries: int = 10,
    ) -> None:
        from openai import OpenAI  # deferred
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — refuse to run without a key"
            )
        self._model_name = model_name
        # Bump from SDK default (2) so Tier-1 RPM/TPM windows on gpt-4o don't
        # silently eat rows — the SDK backs off on the Retry-After header.
        self._client = OpenAI(api_key=resolved_key, max_retries=max_retries)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
