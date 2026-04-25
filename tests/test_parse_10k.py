"""Tests for src.parse_10k.

Two flavors:
  1. Synthetic HTML with TOC + sections — exercises section boundary detection
     without depending on any downloaded filing.
  2. Live spot-check on AAPL FY2023 if it has been downloaded; otherwise skipped.
     This covers the real-world quirks the synthetic test can't.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parse_10k import (
    TARGET_ITEMS,
    drop_table_of_contents,
    extract_sections,
    html_to_text,
    parse_filing,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AAPL_FY2023_HTML = REPO_ROOT / "data" / "10k" / "raw" / "AAPL" / "FY2023.html"


def _synthetic_10k_html() -> str:
    return """<html><body>
<p>Cover page — Form 10-K for fiscal year 2023.</p>
<p>Table of Contents</p>
<p>Item 1.  Business .................. 3</p>
<p>Item 1A. Risk Factors ............. 10</p>
<p>Item 7.  MD&amp;A ................... 40</p>
<p>Item 7A. Market Risk .............. 60</p>
<p>Item 8.  Financial Statements ..... 65</p>

<h1>Item 1. Business</h1>
<p>""" + ("The company designs and sells widgets. " * 40) + """</p>

<h1>Item 1A. Risk Factors</h1>
<p>""" + ("Our business is subject to risks including supply chain disruption. " * 40) + """</p>

<h1>Item 2. Properties</h1>
<p>We lease office space in several cities — skip me.</p>

<h1>Item 7. Management's Discussion and Analysis</h1>
<p>""" + ("Fiscal 2023 revenue grew 5% year-over-year to $100M. " * 40) + """</p>

<h1>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</h1>
<p>""" + ("We are exposed to foreign currency risk across USD, EUR, and JPY. " * 40) + """</p>

<h1>Item 8. Financial Statements and Supplementary Data</h1>
<p>""" + ("Note 1 — Summary of significant accounting policies. " * 40) + """</p>

<h1>Item 9. Changes in Accountants</h1>
<p>None — we should NOT capture this.</p>
</body></html>"""


def test_html_to_text_preserves_paragraph_breaks() -> None:
    html = "<p>First.</p><p>Second.</p>"
    text = html_to_text(html)
    assert "First." in text and "Second." in text
    # Each <p> should produce at least one newline separator
    assert text.count("\n") >= 1


def test_drop_table_of_contents_trims_preamble() -> None:
    # Two occurrences of "Item 1." → keep only from the second onward
    text = "Item 1. Business ... 3\nItem 1A. ...\nlater:\nItem 1. Business\nReal content."
    trimmed = drop_table_of_contents(text)
    assert "Real content." in trimmed
    assert trimmed.count("Item 1.") == 1


def test_extract_sections_on_synthetic_10k() -> None:
    html = _synthetic_10k_html()
    text = html_to_text(html)
    text = drop_table_of_contents(text)
    sections = extract_sections(text)

    # All target items captured
    for item in TARGET_ITEMS:
        assert item in sections, f"missing item {item}"
        assert len(sections[item]) > 100, f"item {item} too short: {len(sections[item])}"

    # Content sanity: each section contains its distinctive phrase
    assert "designs and sells widgets" in sections["1"]
    assert "supply chain disruption" in sections["1A"]
    assert "Fiscal 2023 revenue grew" in sections["7"]
    assert "foreign currency risk" in sections["7A"]
    assert "significant accounting policies" in sections["8"]

    # Non-target items must NOT leak in
    assert "lease office space" not in sections["1A"]  # Item 2 was between 1A and 7
    assert "Changes in Accountants" not in sections["8"]  # Item 9 ends Item 8


@pytest.mark.skipif(not AAPL_FY2023_HTML.exists(), reason="AAPL FY2023 not downloaded")
def test_aapl_fy2023_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect output to tmp so the test doesn't clobber real section files
    from src import parse_10k
    monkeypatch.setattr(parse_10k, "SECTIONS_DIR", tmp_path)
    stats = parse_filing("AAPL", 2023, AAPL_FY2023_HTML)

    assert stats.parse_success, f"AAPL FY2023 parse failed, missing: {stats.missing_items}"
    # Items 1, 1A, 7, 7A, 8 should all have substantial content
    by_item = {s.item: s.char_count for s in stats.sections}
    for item in ("1", "1A", "7", "8"):
        assert by_item[item] > 5000, f"item {item} suspiciously short: {by_item[item]}"

    # Distinctive content from AAPL 10-K
    item1_text = (tmp_path / "AAPL" / "FY2023" / "item_1.txt").read_text()
    assert "iPhone" in item1_text or "smartphones" in item1_text

    # _stats.json round-trip
    loaded = json.loads((tmp_path / "AAPL" / "FY2023" / "_stats.json").read_text())
    assert loaded["parse_success"] is True
