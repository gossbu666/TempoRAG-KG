"""Parse downloaded 10-K HTML into clean section text.

Extracts only the temporal-rich sections (per docs/10k_scoping.md §2):
    Item 1  — Business
    Item 1A — Risk Factors
    Item 7  — Management's Discussion & Analysis (MD&A)
    Item 7A — Quantitative/Qualitative Disclosures About Market Risk
    Item 8  — Financial Statements and Supplementary Data (notes)

Everything else (Items 2–6, 9–14, 15, Part III, cover page, signatures, exhibits)
is intentionally dropped because it is boilerplate or non-temporal.

Output:
    data/10k/sections/<ticker>/FY<year>/<item>.txt  — UTF-8 plain text per section
    data/10k/sections/<ticker>/FY<year>/_stats.json — per-filing parse stats

A filing that fails to produce text for >= 2 expected sections is flagged as a
parse failure and logged; no silent skips. Success rate is reported in the
per-filing manifest written at data/10k/parse_manifest.json.

Parsing strategy:
    1. BeautifulSoup with the `lxml` backend strips HTML to text while preserving
       paragraph breaks from block-level elements.
    2. Table of Contents at the start is removed: we find the *second* occurrence
       of "Item 1." (the first is the TOC entry, the second is the actual heading).
    3. Section boundaries: we locate each target `Item N` heading and capture
       everything until the next known Item heading. A filing rarely follows a
       perfectly linear order (e.g. some 10-Ks place Item 7A after a sub-Item);
       the "next heading" rule tolerates this.
    4. Two non-canonical filing styles are handled specially:
       - **Cross-reference style** (INTC): the filing uses descriptive titles
         ("Risk Factors", "Management's Discussion and Analysis") as real
         headings; the canonical "Item N." labels appear only in an end-of-doc
         cross-reference index. Detected via: first `Item 1` match is past 70%
         of the document. Fallback: anchor on descriptive titles, using the
         second standalone-line occurrence (first is in the TOC).
       - **Item 8 pointer** (ORCL/NVDA): Item 8 body contains only a pointer
         like "See Part IV, Item 15" and the real financial statements are
         appended under Item 15. Detected via: Item 8 body < threshold AND
         contains "Item 15". Fallback: substitute Item 15 body as Item 8.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Many 10-K filings are served as XHTML; lxml parses them as XML and emits a
# noisy warning. Silence it — we rely on the lenient HTML extraction path and
# have tests that cover the real filings.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "10k"
RAW_DIR = DATA_DIR / "raw"
SECTIONS_DIR = DATA_DIR / "sections"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PARSE_MANIFEST_PATH = DATA_DIR / "parse_manifest.json"
PARSE_FAILURES_PATH = DATA_DIR / "parse_failures.jsonl"

TARGET_ITEMS: tuple[str, ...] = ("1", "1A", "7", "7A", "8")

# Ordered list of every Item heading we care about as a boundary marker.
# A heading can start a *target* section or just act as the end of the preceding one.
ALL_ITEM_LABELS: tuple[str, ...] = (
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16",
)

# Matches headings like:
#   Item 1.
#   Item 1 —
#   ITEM 1A
#   Item  1A.  Risk Factors
# Anchored to line start (after the soup-to-text conversion inserts newlines).
def _heading_re(label: str) -> re.Pattern:
    # Escape any alphanumeric suffix ("1A") and require a word boundary after it.
    esc = re.escape(label)
    return re.compile(
        rf"(?mi)^\s*item\s+{esc}\b[\s\.\-—–:]*",
    )


HEADING_PATTERNS: dict[str, re.Pattern] = {lbl: _heading_re(lbl) for lbl in ALL_ITEM_LABELS}

# Descriptive titles used as real section headings in non-canonical filings
# (currently INTC FY2022-FY2024). Each target item is anchored on the 2nd
# standalone-line occurrence of the first matching phrase (1st is the TOC entry).
# Ordering within a tuple is fallback-preference, not Item order.
# Tunables for _find_title_anchor:
#   _CLUSTER_GAP: two line-start matches closer than this are treated as one
#       visual heading (INTC wraps "Item 1A. Risk Factors" across two lines, so
#       "Risk Factors" matches twice within ~40 chars).
#   _TOC_END_CUTOFF: anything before this offset is assumed to be cover page /
#       TOC region. Chosen from inspection: all 10-K TOCs in the corpus end by
#       ~10k chars; 15k is a safe margin.
_CLUSTER_GAP = 200
_TOC_END_CUTOFF = 15000

TITLE_ANCHORS: dict[str, tuple[str, ...]] = {
    "1":  ("Fundamentals of Our Business",),
    "1A": ("Risk Factors",),
    "7":  (
        "Management's Discussion and Analysis",
        "Management\u2019s Discussion and Analysis",  # curly apostrophe variant
    ),
    "7A": ("Quantitative and Qualitative Disclosures About Market Risk",),
    "8":  (
        "Index to Consolidated Financial Statements",
        "Consolidated Financial Statements",
    ),
}

# Boilerplate / navigational lines we want to scrub post-extraction.
BOILERPLATE_LINE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*index\s+to\s+consolidated\s+financial\s+statements\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # standalone page numbers
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),  # "- 42 -" page markers
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("parse_10k")


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class SectionStats:
    item: str
    char_count: int
    line_count: int
    written_to: str


@dataclass
class FilingParseStats:
    ticker: str
    fiscal_year: int
    source_html: str
    sections: list[SectionStats]
    missing_items: list[str]
    parse_success: bool


def html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving paragraph structure."""
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious non-content before text extraction.
    for tag in soup(["script", "style", "ix:header", "ix:references", "ix:resources", "ix:hidden"]):
        tag.decompose()

    # Ensure block-level elements create newlines in the output.
    # BeautifulSoup's get_text(separator="\n") is close enough once <br> and block tags
    # are converted; we manually insert \n after commonly-used block tags.
    for block in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
        block.append("\n")

    text = soup.get_text(separator=" ")
    # Normalize whitespace without destroying paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_table_of_contents(text: str) -> str:
    """Heuristic: the real Item 1 heading is typically the second occurrence.

    If we find ≥ 2 matches for "Item 1.", drop everything before the second one —
    that removes the cover page + TOC in one step. If only one match, return as-is
    (the filing may lack a TOC).
    """
    matches = list(HEADING_PATTERNS["1"].finditer(text))
    if len(matches) >= 2:
        return text[matches[1].start():]
    return text


def _find_all_item_offsets(text: str) -> list[tuple[str, int]]:
    """Return [(label, char_offset), ...] sorted by offset, keeping only the FIRST
    hit for each label (by this point TOC has been trimmed, so the first hit is
    the actual heading for that item)."""
    first_hits: dict[str, int] = {}
    for label, pattern in HEADING_PATTERNS.items():
        m = pattern.search(text)
        if m:
            first_hits[label] = m.start()
    return sorted(first_hits.items(), key=lambda kv: kv[1])


def _is_cross_reference_style(text: str) -> bool:
    """Detect filings like INTC that expose Item-N labels only in an
    end-of-document cross-reference index.

    Signal: the first `Item 1` match sits past 70% of the document. A canonical
    filing's first post-TOC `Item 1` heading sits in the first 10-20%; anything
    past 70% means the match is actually inside a trailing reference table.
    """
    m = HEADING_PATTERNS["1"].search(text)
    if m is None:
        return True
    return m.start() > 0.7 * len(text)


def _find_title_anchor(text: str, phrases: tuple[str, ...]) -> int | None:
    """Return the offset of the real (body) occurrence of a section title.

    Strategy: try each candidate phrase in preference order. Collect line-start
    matches (to avoid in-paragraph mentions), collapse matches within
    `_CLUSTER_GAP` chars of each other into a single cluster (a wrapped layout
    often produces two near-adjacent line-start matches for one visual heading),
    then pick:
      - The 2nd cluster if ≥ 2 clusters (1st = TOC entry, 2nd = body heading).
      - The 1st cluster if it sits past the TOC region (single-cluster phrases
        like "Index to Consolidated Financial Statements" that appear only in
        the body — their TOC counterpart is a different phrase).
    """
    for phrase in phrases:
        pattern = re.compile(rf"(?mi)^\s*{re.escape(phrase)}\b")
        offsets = [m.start() for m in pattern.finditer(text)]
        if not offsets:
            continue
        clusters = [offsets[0]]
        for off in offsets[1:]:
            if off - clusters[-1] > _CLUSTER_GAP:
                clusters.append(off)
        if len(clusters) >= 2:
            return clusters[1]
        if clusters[0] > _TOC_END_CUTOFF:
            return clusters[0]
    return None


def _extract_by_title(text: str) -> dict[str, str]:
    """Fallback section extraction using descriptive-title anchors.

    Section boundaries are derived from the offset order of the anchors, not
    the canonical Item order — INTC's 10-K literally presents sections as
    1 → 7 → 1A → 7A → 8, not 1 → 1A → 7 → 7A → 8.
    """
    anchors: list[tuple[str, int]] = []
    for label, phrases in TITLE_ANCHORS.items():
        offset = _find_title_anchor(text, phrases)
        if offset is not None:
            anchors.append((label, offset))
    anchors.sort(key=lambda x: x[1])

    out: dict[str, str] = {}
    for i, (label, start) in enumerate(anchors):
        end = anchors[i + 1][1] if i + 1 < len(anchors) else len(text)
        body = text[start:end]
        newline_pos = body.find("\n")
        if newline_pos != -1:
            body = body[newline_pos + 1:]
        out[label] = strip_boilerplate(body).strip()
    return out


_POINTER_PHRASES = re.compile(
    r"(?i)\bitem\s+15\b"
    r"|\bsubmitted\s+as\s+a\s+separate\s+section\b"
    r"|\b(?:set\s+forth|included|contained)\s+in\s+"
    r"(?:our\s+)?(?:consolidated\s+)?financial\s+statements\b"
)


def _is_item8_pointer(body: str) -> bool:
    """Detect Item 8 body that redirects elsewhere instead of containing the
    financial statements.

    Observed verbiage across our 10 tickers:
      - ORCL: "The response to this item is submitted as a separate section of
        this Annual Report. See Part IV, Item 15."
      - NVDA: "The information required by this Item is set forth in our
        Consolidated Financial Statements and Notes thereto included in this
        Annual Report..."
    Both have bodies < 500 chars and either reference "Item 15" or the phrase
    "set forth/included in ... financial statements".
    """
    if len(body) >= 500:
        return False
    return bool(_POINTER_PHRASES.search(body))


def _item15_body(text: str, ordered_items: list[tuple[str, int]]) -> str | None:
    """Extract Item 15's full body (used as Item 8 for ORCL/NVDA-style filings)."""
    idx = next((i for i, (lbl, _) in enumerate(ordered_items) if lbl == "15"), None)
    if idx is None:
        return None
    start = ordered_items[idx][1]
    end = ordered_items[idx + 1][1] if idx + 1 < len(ordered_items) else len(text)
    body = text[start:end]
    newline_pos = body.find("\n")
    if newline_pos != -1:
        body = body[newline_pos + 1:]
    return strip_boilerplate(body).strip()


def extract_sections(text: str) -> dict[str, str]:
    """Given TOC-trimmed text, split by item headings and return {target_item: body_text}.

    Target items keep their body; non-target items act only as end-boundaries.
    Two non-canonical paths: cross-reference style (INTC) uses descriptive-title
    anchors; pointer-style Item 8 (ORCL/NVDA) is replaced with Item 15 body.
    """
    if _is_cross_reference_style(text):
        return _extract_by_title(text)

    ordered = _find_all_item_offsets(text)
    out: dict[str, str] = {}
    for i, (label, start) in enumerate(ordered):
        if label not in TARGET_ITEMS:
            continue
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        body = text[start:end]
        # Strip the heading line itself; keep the body after the first newline.
        newline_pos = body.find("\n")
        if newline_pos != -1:
            body = body[newline_pos + 1:]
        out[label] = strip_boilerplate(body).strip()

    if "8" in out and _is_item8_pointer(out["8"]):
        replacement = _item15_body(text, ordered)
        if replacement and len(replacement) >= 500:
            out["8"] = replacement
    return out


def strip_boilerplate(text: str) -> str:
    """Drop TOC/page-number lines; collapse repeated blank lines."""
    kept: list[str] = []
    for line in text.split("\n"):
        if any(pat.match(line) for pat in BOILERPLATE_LINE_PATTERNS):
            continue
        kept.append(line)
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out)


def parse_filing(ticker: str, fiscal_year: int, html_path: Path) -> FilingParseStats:
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    text = html_to_text(raw)
    text = drop_table_of_contents(text)
    sections = extract_sections(text)

    out_dir = SECTIONS_DIR / ticker / f"FY{fiscal_year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    section_stats: list[SectionStats] = []
    for item in TARGET_ITEMS:
        body = sections.get(item, "")
        out_path = out_dir / f"item_{item}.txt"
        out_path.write_text(body, encoding="utf-8")
        section_stats.append(SectionStats(
            item=item,
            char_count=len(body),
            line_count=body.count("\n") + (1 if body else 0),
            written_to=_rel_or_abs(out_path),
        ))

    # A section is considered "present" if it has at least a few hundred characters
    # of real content. Threshold chosen to tolerate short Item 7A blocks in AAPL
    # but catch totally-empty extractions.
    PRESENCE_THRESHOLD = 500
    missing = [s.item for s in section_stats if s.char_count < PRESENCE_THRESHOLD]
    parse_success = len(missing) <= 1  # tolerate 1 missing (often Item 7A for some filings)

    stats = FilingParseStats(
        ticker=ticker,
        fiscal_year=fiscal_year,
        source_html=_rel_or_abs(html_path),
        sections=section_stats,
        missing_items=missing,
        parse_success=parse_success,
    )
    stats_path = out_dir / "_stats.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2))
    return stats


def main() -> int:
    if not MANIFEST_PATH.exists():
        log.error("manifest missing at %s — run scripts/download_10k.py first", MANIFEST_PATH)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text())

    all_stats: list[FilingParseStats] = []
    failures: list[dict] = []

    for entry in manifest.get("filings", []):
        ticker = entry["ticker"]
        fy = entry["fiscal_year"]
        html_path = REPO_ROOT / entry["local_path"]
        if not html_path.exists():
            log.error("html missing for %s FY%d at %s", ticker, fy, html_path)
            failures.append({"ticker": ticker, "fiscal_year": fy, "error": "html file missing"})
            continue
        try:
            stats = parse_filing(ticker, fy, html_path)
            all_stats.append(stats)
            marker = "OK " if stats.parse_success else "BAD"
            log.info(
                "%s %s FY%d: %s, missing=%s",
                marker, ticker, fy,
                " ".join(f"{s.item}={s.char_count}" for s in stats.sections),
                stats.missing_items,
            )
            if not stats.parse_success:
                failures.append({
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "error": f"too many missing items: {stats.missing_items}",
                })
        except Exception as e:  # noqa: BLE001
            log.exception("parse crashed for %s FY%d", ticker, fy)
            failures.append({"ticker": ticker, "fiscal_year": fy, "error": f"exception: {e}"})

    parse_manifest = {
        "source_manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "filings": [asdict(s) for s in all_stats],
        "summary": {
            "total": len(all_stats),
            "successful": sum(1 for s in all_stats if s.parse_success),
            "failed": len(failures),
        },
    }
    PARSE_MANIFEST_PATH.write_text(json.dumps(parse_manifest, indent=2, sort_keys=True))
    with PARSE_FAILURES_PATH.open("w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail, sort_keys=True) + "\n")

    s = parse_manifest["summary"]
    log.info("parse done: %d successful, %d failed, out of %d", s["successful"], s["failed"], s["total"])
    log.info("parse manifest → %s", PARSE_MANIFEST_PATH)
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
