"""Tests for src.kg_extract — prompt rendering, cache behavior, response parsing.

Never hits the network: a FakeClient returns canned responses. See
src/kg_extract.py for the pipeline this exercises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cache import Cache
from src.kg_extract import (
    ALLOWED_TEMPORAL_TYPES,
    DEFAULT_COMPANY_NAMES,
    extract_triples,
    format_chunk_metadata,
    parse_response,
    render_prompt,
)


# ------------------------- fixtures ------------------------------------------

_CHUNK_TEXT = "During fiscal 2022, the Company recognized total net sales of $394.3 billion."

_VALID_TRIPLE = {
    "subject": "Apple Inc.",
    "predicate": "net_sales_usd_billion",
    "object": 394.3,
    "valid_from": 2022,
    "valid_to": 2022,
    "confidence": 0.98,
    "evidence": "net sales of $394.3 billion",
    "metadata": {"temporal_type": "explicit"},
}


def _chunk(
    *,
    chunk_id: str = "AAPL_FY2022_item7_000",
    ticker: str = "AAPL",
    fy: int = 2022,
    item: str = "7",
    text: str = "During fiscal 2022, the Company recognized total net sales of $394.3 billion.",
    filing_date: str = "2022-10-28",
    period_of_report: str = "2022-09-24",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "fy": fy,
        "item": item,
        "text": text,
        "filing_date": filing_date,
        "period_of_report": period_of_report,
        "token_count": 25,
        "sha256": "0" * 64,
    }


_MINIMAL_TEMPLATE = (
    "PROMPT HEADER\n"
    "CHUNK METADATA:\n{{CHUNK_METADATA}}\n"
    "PASSAGE:\n{{PASSAGE}}\n"
    "END"
)


class FakeClient:
    """Deterministic LLM substitute. Records the last prompt seen."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, float]] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append((prompt, temperature))
        return self.response_text


@pytest.fixture()
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache")


# ------------------------- format_chunk_metadata -----------------------------


def test_format_chunk_metadata_contains_all_fields() -> None:
    block = format_chunk_metadata(_chunk())
    assert "ticker:           AAPL" in block
    assert "company_name:     Apple Inc." in block
    assert "filing_date:      2022-10-28" in block
    assert "period_of_report: 2022-09-24" in block
    assert "fiscal_year:      2022" in block
    assert "item:             7" in block


def test_format_chunk_metadata_falls_back_to_ticker_when_unmapped() -> None:
    chunk = _chunk(ticker="XYZW", chunk_id="XYZW_FY2022_item7_000")
    block = format_chunk_metadata(chunk, company_name_map={})
    # Unknown tickers should render the ticker itself as the company name
    # rather than crashing the whole extraction run on a config gap.
    assert "company_name:     XYZW" in block


def test_format_chunk_metadata_uses_default_name_map() -> None:
    for ticker, expected_name in DEFAULT_COMPANY_NAMES.items():
        chunk = _chunk(ticker=ticker, chunk_id=f"{ticker}_FY2022_item7_000")
        block = format_chunk_metadata(chunk)
        assert f"company_name:     {expected_name}" in block


# ------------------------- render_prompt -------------------------------------


def test_render_prompt_substitutes_both_placeholders() -> None:
    out = render_prompt(_MINIMAL_TEMPLATE, _chunk())
    assert "{{CHUNK_METADATA}}" not in out
    assert "{{PASSAGE}}" not in out
    assert "During fiscal 2022" in out
    assert "ticker:           AAPL" in out


def test_render_prompt_rejects_template_missing_placeholder() -> None:
    bad = "PROMPT HEADER\nCHUNK METADATA:\n{{CHUNK_METADATA}}\nEND"  # no {{PASSAGE}}
    with pytest.raises(ValueError):
        render_prompt(bad, _chunk())


def test_render_prompt_on_real_template() -> None:
    """The actual prompt file on disk must still render cleanly."""
    tmpl = Path("prompts/extract_v1.txt").read_text(encoding="utf-8")
    out = render_prompt(tmpl, _chunk())
    assert "{{CHUNK_METADATA}}" not in out
    assert "{{PASSAGE}}" not in out


# ------------------------- parse_response ------------------------------------


def test_parse_response_valid() -> None:
    raw = json.dumps({"triples": [_VALID_TRIPLE]})
    triples, errors = parse_response(raw, _CHUNK_TEXT)
    assert errors == []
    assert triples == [_VALID_TRIPLE]


def test_parse_response_strips_json_code_fence() -> None:
    raw = "```json\n" + json.dumps({"triples": [_VALID_TRIPLE]}) + "\n```"
    triples, errors = parse_response(raw, _CHUNK_TEXT)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_strips_bare_code_fence() -> None:
    raw = "```\n" + json.dumps({"triples": [_VALID_TRIPLE]}) + "\n```"
    triples, errors = parse_response(raw, _CHUNK_TEXT)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_rejects_invalid_json() -> None:
    triples, errors = parse_response("{not valid json", _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "JSONDecodeError" in errors[0]


def test_parse_response_rejects_missing_triples_key() -> None:
    triples, errors = parse_response(json.dumps({"something_else": []}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "missing 'triples'" in errors[0]


def test_parse_response_rejects_triples_not_a_list() -> None:
    triples, errors = parse_response(json.dumps({"triples": "not a list"}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "array" in errors[0]


def test_parse_response_rejects_triple_missing_keys() -> None:
    bad = {"subject": "X", "predicate": "p", "object": "y"}  # missing most fields
    triples, errors = parse_response(json.dumps({"triples": [bad]}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "missing keys" in errors[0]


def test_parse_response_rejects_bad_temporal_type() -> None:
    bad = dict(_VALID_TRIPLE)
    bad["metadata"] = {"temporal_type": "some_bogus_value"}
    triples, errors = parse_response(json.dumps({"triples": [bad]}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "temporal_type" in errors[0]


def test_parse_response_rejects_non_int_years() -> None:
    bad = dict(_VALID_TRIPLE)
    bad["valid_from"] = "2022"  # string, not int
    triples, errors = parse_response(json.dumps({"triples": [bad]}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "valid_from" in errors[0]


def test_parse_response_rejects_confidence_out_of_range() -> None:
    bad = dict(_VALID_TRIPLE)
    bad["confidence"] = 1.5
    triples, errors = parse_response(json.dumps({"triples": [bad]}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "confidence" in errors[0]


def test_parse_response_accepts_all_allowed_temporal_types() -> None:
    for tt in ALLOWED_TEMPORAL_TYPES:
        good = dict(_VALID_TRIPLE)
        good["metadata"] = {"temporal_type": tt}
        triples, errors = parse_response(json.dumps({"triples": [good]}), _CHUNK_TEXT)
        assert errors == [], (tt, errors)
        assert len(triples) == 1


def test_parse_response_empty_triples_list_is_valid() -> None:
    # The LLM correctly emitted zero triples for a purely boilerplate chunk.
    # That's a legitimate outcome, not a parse error.
    triples, errors = parse_response(json.dumps({"triples": []}), _CHUNK_TEXT)
    assert errors == []
    assert triples == []


def test_parse_response_keeps_valid_triples_when_some_fail() -> None:
    """Per-triple validation: one bad triple must not discard the good ones.
    Before this change, a single failure in a batch of many threw away the
    entire chunk — in P1 pilot that meant losing 25 valid triples from an
    Item 8 table because one fabricated value rejected the whole batch.
    """
    good = _VALID_TRIPLE
    bad = dict(_VALID_TRIPLE)
    bad["object"] = 999.9  # fabricated — not in chunk
    raw = json.dumps({"triples": [good, bad, good]})
    triples, errors = parse_response(raw, _CHUNK_TEXT)
    assert triples == [good, good], "valid triples must survive even if a sibling fails"
    assert len(errors) == 1
    assert "triple[1]" in errors[0] and "object not found" in errors[0]


# ------------------------- hallucination guard (object-based) ----------------


def test_parse_response_rejects_object_not_in_chunk() -> None:
    """A fabricated numeric object — one the chunk has no textual form of —
    must be rejected. This is the core hallucination guard: the CLAIM
    entering the KG is what we verify, not the model's evidence string.
    """
    bad = dict(_VALID_TRIPLE)
    bad["object"] = 999.9  # chunk has 394.3, not 999.9
    triples, errors = parse_response(json.dumps({"triples": [bad]}), _CHUNK_TEXT)
    assert triples == []
    assert len(errors) == 1 and "object not found" in errors[0]


def test_parse_response_accepts_object_across_pdf_table_cells() -> None:
    """10-K tables split each cell onto its own line, leaving gaps like
    `$\\n\\n25,821`. The model's claim (object=25821) must still verify
    against the whitespace-stripped chunk.
    """
    chunk = "Americas\n\n$\n\n25,821\n\n 25%\n\n 26%"
    good = dict(_VALID_TRIPLE)
    good["object"] = 25821
    good["evidence"] = "anything — evidence is audit-only"
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_accepts_object_without_currency_prefix() -> None:
    """PDF tables put `$` only on the header row, not every cell. Model
    infers the column is dollars and emits object=9930 for an EMEA row
    that literally reads `9,930`. Object-based matching finds the number
    regardless of how the chunk formats the currency marker.
    """
    chunk = "EMEA\n\n9,930\n\n -1%\n\n 6%"
    good = dict(_VALID_TRIPLE)
    good["object"] = 9930
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_accepts_percentage_as_decimal_fraction() -> None:
    """Models sometimes normalize `"64%"` to object=0.64 (fraction form).
    The chunk has `"64%"`, not `"0.64"`. Candidate-form matching must
    accept both representations of the same value.
    """
    chunk = "Total Margin 64%"
    good = dict(_VALID_TRIPLE)
    good["object"] = 0.64
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_accepts_single_digit_percentages() -> None:
    """Financial tables freely use `"6%"`, `"-1%"`, `"8%"`. With
    object-based verification the short percent no longer trips a
    length guard — object=6 is matched against `"6%"` in the chunk.
    """
    chunk = "Constant currency change 6%"
    good = dict(_VALID_TRIPLE)
    good["object"] = 6
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_accepts_comma_separated_number() -> None:
    """Object=9930 must match chunk text `9,930` (comma thousands sep)
    just as readily as `9930`. Candidate list covers both forms.
    """
    chunk = "Operating expenses 9,930 million"
    good = dict(_VALID_TRIPLE)
    good["object"] = 9930
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_accepts_string_object_in_chunk() -> None:
    """String objects (e.g., a proper noun) are matched by substring
    against the whitespace-stripped chunk — same rule as numerics.
    """
    chunk = "The Americas segment contributed most revenue."
    good = dict(_VALID_TRIPLE)
    good["object"] = "Americas"
    triples, errors = parse_response(json.dumps({"triples": [good]}), chunk)
    assert errors == []
    assert len(triples) == 1


def test_parse_response_ignores_evidence_content() -> None:
    """Evidence is an audit trail, not a gate. A paraphrased or even
    fabricated evidence string does not block the triple as long as
    the object verifies against the chunk. (Object is what enters the KG.)
    """
    good = dict(_VALID_TRIPLE)
    # object=394.3 is in _CHUNK_TEXT; evidence text is wrong but audit-only.
    good["evidence"] = "Tim Cook became CEO in 2011"
    triples, errors = parse_response(json.dumps({"triples": [good]}), _CHUNK_TEXT)
    assert errors == []
    assert len(triples) == 1


# ------------------------- extract_triples end-to-end ------------------------


def test_extract_triples_miss_then_hit(cache: Cache) -> None:
    raw = json.dumps({"triples": [_VALID_TRIPLE]})
    client = FakeClient(raw)
    chunk = _chunk()

    result1 = extract_triples(chunk, _MINIMAL_TEMPLATE, client, cache)
    assert result1.cache_hit is False
    assert result1.parse_errors == []
    assert result1.triples == [_VALID_TRIPLE]
    assert len(client.calls) == 1

    result2 = extract_triples(chunk, _MINIMAL_TEMPLATE, client, cache)
    assert result2.cache_hit is True
    assert result2.triples == [_VALID_TRIPLE]
    assert len(client.calls) == 1, "cache hit must not invoke client"


def test_extract_triples_caches_raw_not_parsed(cache: Cache) -> None:
    raw = json.dumps({"triples": [_VALID_TRIPLE]})
    client = FakeClient(raw)
    result = extract_triples(_chunk(), _MINIMAL_TEMPLATE, client, cache)

    # The on-disk cache entry must store `response` (raw text), not parsed triples.
    # This keeps the cache reusable if the parser gets stricter later.
    key = cache.key_for("gemini-1.5-flash", render_prompt(_MINIMAL_TEMPLATE, _chunk()), {"temperature": 0.0})
    entry = cache.get(key)
    assert entry is not None
    assert "response" in entry and entry["response"] == raw
    assert "triples" not in entry


def test_extract_triples_cache_busts_when_prompt_changes(cache: Cache) -> None:
    client = FakeClient(json.dumps({"triples": [_VALID_TRIPLE]}))
    extract_triples(_chunk(), _MINIMAL_TEMPLATE, client, cache)

    revised_template = _MINIMAL_TEMPLATE.replace("PROMPT HEADER", "PROMPT HEADER v2")
    extract_triples(_chunk(), revised_template, client, cache)
    assert len(client.calls) == 2, "changed prompt template must bypass cached response"


def test_extract_triples_cache_busts_when_chunk_changes(cache: Cache) -> None:
    client = FakeClient(json.dumps({"triples": [_VALID_TRIPLE]}))
    extract_triples(_chunk(chunk_id="a"), _MINIMAL_TEMPLATE, client, cache)
    other = _chunk(chunk_id="b", text="A completely different passage.")
    extract_triples(other, _MINIMAL_TEMPLATE, client, cache)
    assert len(client.calls) == 2, "different chunk text must bypass cached response"


def test_extract_triples_cache_survives_parse_error(cache: Cache) -> None:
    """A malformed LLM response is still cached so we don't pay for it twice.
    Fatal parse errors surface on `parse_errors`, not as an exception.
    """
    client = FakeClient("this is not valid JSON at all")
    result1 = extract_triples(_chunk(), _MINIMAL_TEMPLATE, client, cache)
    assert result1.parse_errors != []
    assert result1.triples == []

    result2 = extract_triples(_chunk(), _MINIMAL_TEMPLATE, client, cache)
    assert result2.cache_hit is True, "bad response must still be cached"
    assert result2.parse_errors != []
    assert len(client.calls) == 1, "re-invoking a known-bad chunk must not re-spend money"


def test_extract_triples_does_not_raise_on_bad_json(cache: Cache) -> None:
    client = FakeClient("absolutely not JSON")
    result = extract_triples(_chunk(), _MINIMAL_TEMPLATE, client, cache)
    # Must behave like a structured failure, not an exception.
    assert result.parse_errors != []
    assert result.cache_hit is False


def test_extract_triples_prompt_sent_contains_metadata(cache: Cache) -> None:
    client = FakeClient(json.dumps({"triples": []}))
    chunk = _chunk(ticker="MSFT", fy=2023, chunk_id="MSFT_FY2023_item7_000")
    extract_triples(chunk, _MINIMAL_TEMPLATE, client, cache)
    sent_prompt, temp = client.calls[0]
    assert "ticker:           MSFT" in sent_prompt
    assert "company_name:     Microsoft Corporation" in sent_prompt
    assert "fiscal_year:      2023" in sent_prompt
    assert temp == 0.0
