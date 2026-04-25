"""Download 10-K filings from SEC EDGAR for the 10-ticker corpus.

Target: 45 filings total.
    - Mag5 (AAPL, MSFT, GOOGL, AMZN, META) × FY2019–FY2024 = 30 filings
    - CSCO, ORCL, INTC, NVDA, ADBE × FY2022–FY2024 = 15 filings

The year ranges differ per ticker: Mag5 keeps the full FY2019-2023 corpus from
the original A3a delivery (useful for home-grown retro Qs) and adds FY2024 to
match the FinReflectKG-MultiHop eval window. The 5 new tickers are scoped to
FY2022-2024 only, matching MultiHop's coverage — pulling FY<2022 for them has
no downstream use and would waste SEC quota.

Output layout:
    data/10k/raw/<ticker>/FY<year>.html     — primary 10-K document (HTML)
    data/10k/manifest.json                  — per-filing metadata + success flags
    data/10k/failures.jsonl                 — one line per failed filing (never silent)

Reliability:
    - Respects SEC EDGAR fair-access policy: User-Agent with real contact email,
      rate-limited to ≤5 req/s (well under their 10 req/s cap), single-threaded.
    - Retries on 429/5xx with exponential backoff (3 attempts, base 2s).
    - Submissions index cached per ticker for re-runs at zero network cost.
    - Skips download if the target HTML already exists on disk (idempotent); the
      manifest is rebuilt from disk so we still record already-present filings.

Fiscal year note:
    - AAPL, CSCO, MSFT, ORCL use off-calendar fiscal years; we match by the
      calendar year of `periodOfReport` (SEC's authoritative FY label).
    - INTC, ADBE, NVDA, GOOGL, AMZN, META align closer to calendar year.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "10k"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "_edgar_cache"
MANIFEST_PATH = DATA_DIR / "manifest.json"
FAILURES_PATH = DATA_DIR / "failures.jsonl"

# CIKs must be zero-padded to 10 digits for the submissions endpoint.
TICKERS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "CSCO": "0000858877",
    "ORCL": "0001341439",
    "INTC": "0000050863",
    "NVDA": "0001045810",
    "ADBE": "0000796343",
}

MAG5: frozenset[str] = frozenset({"AAPL", "MSFT", "GOOGL", "AMZN", "META"})

# Per-ticker year scope. Mag5 = FY2019-2024 (pre-2022 kept for retro home-grown
# Qs); the 5 tech expansion tickers = FY2022-2024 only (MultiHop eval window).
TICKER_FYS: dict[str, tuple[int, ...]] = {
    ticker: (2019, 2020, 2021, 2022, 2023, 2024) if ticker in MAG5
    else (2022, 2023, 2024)
    for ticker in TICKERS
}

RATE_LIMIT_SLEEP_S = 0.2  # 5 req/s, under SEC's 10 req/s cap
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_10k")


@dataclass
class FilingRecord:
    ticker: str
    cik: str
    fiscal_year: int
    accession_no: str
    period_of_report: str  # YYYY-MM-DD
    filing_date: str       # YYYY-MM-DD
    primary_document: str  # filename inside the filing folder
    document_url: str      # fully qualified URL
    local_path: str        # relative to repo root
    size_bytes: int
    sha256: str


def build_user_agent() -> str:
    """SEC requires identification. Uses contact email from env, falling back to a sentinel."""
    load_dotenv(REPO_ROOT / ".env")
    email = os.getenv("SEC_CONTACT_EMAIL") or os.getenv("CONTACT_EMAIL") or "bsupanutkom@gmail.com"
    return f"TempoRAG-KG Research (AIT) {email}"


def make_session() -> requests.Session:
    s = requests.Session()
    # No Host header on the session — requests derives it per-URL. Cross-subdomain
    # requests (data.sec.gov vs www.sec.gov) would otherwise get the wrong Host.
    s.headers.update({
        "User-Agent": build_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def polite_get(session: requests.Session, url: str) -> requests.Response:
    """GET with SEC-aware rate limit and retries. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            time.sleep(RATE_LIMIT_SLEEP_S)
            resp = session.get(url, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} from {url}")
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            last_exc = e
            wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
            log.warning("attempt %d/%d failed for %s: %s; sleep %.1fs", attempt, MAX_ATTEMPTS, url, e, wait)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _fetch_and_cache_json(session: requests.Session, url: str, cache_path: Path) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    log.info("fetching %s", url)
    resp = polite_get(session, url)
    data = resp.json()
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def load_submissions(session: requests.Session, cik: str) -> dict:
    """Fetch + cache the primary submissions JSON for a CIK."""
    return _fetch_and_cache_json(
        session,
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        CACHE_DIR / f"submissions_{cik}.json",
    )


def load_archived_submissions(session: requests.Session, file_name: str) -> dict:
    """Fetch + cache one of the paginated submission archives (for older filings)."""
    return _fetch_and_cache_json(
        session,
        f"https://data.sec.gov/submissions/{file_name}",
        CACHE_DIR / file_name,
    )


def _iter_filing_rows(index: dict):
    """Yield (form, accession, filing_date, period_of_report, primary_document) tuples.

    Works on both the `filings.recent` shape (primary submissions) and the archive
    shape (paginated files referenced by `filings.files[]`), which has the same
    column-oriented structure at the top level.
    """
    if "filings" in index and "recent" in index["filings"]:
        table = index["filings"]["recent"]
    else:
        table = index
    forms = table.get("form", [])
    accessions = table.get("accessionNumber", [])
    filing_dates = table.get("filingDate", [])
    periods = table.get("reportDate", [])
    primary_docs = table.get("primaryDocument", [])
    return zip(forms, accessions, filing_dates, periods, primary_docs)


def _match_10k(index: dict, target_fy: int) -> dict | None:
    for form, accession, fdate, pdate, pdoc in _iter_filing_rows(index):
        if form != "10-K" or not pdate:
            continue
        try:
            year = int(pdate.split("-")[0])
        except (ValueError, IndexError):
            continue
        if year == target_fy:
            return {
                "accession_no": accession,
                "filing_date": fdate,
                "period_of_report": pdate,
                "primary_document": pdoc,
            }
    return None


def pick_10k_for_fy(session: requests.Session, submissions: dict, target_fy: int) -> dict | None:
    """Find the 10-K whose period-of-report year == target_fy.

    Looks first in `filings.recent`; if absent, walks the paginated archives
    listed in `filings.files[]` (older filings live there).
    """
    hit = _match_10k(submissions, target_fy)
    if hit is not None:
        return hit

    for archive_meta in submissions.get("filings", {}).get("files", []):
        # Each archive has filingFrom / filingTo dates — skip archives that can't contain
        # a 10-K for this FY (filing dates run filingFrom..filingTo; 10-Ks usually file
        # within ~4 months of FY end).
        archive = load_archived_submissions(session, archive_meta["name"])
        hit = _match_10k(archive, target_fy)
        if hit is not None:
            return hit
    return None


def build_document_url(cik: str, accession_no: str, primary_document: str) -> str:
    """EDGAR archive URL. CIK must NOT be zero-padded here; accession stripped of dashes."""
    cik_int = str(int(cik))
    acc_nodashes = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodashes}/{primary_document}"


def download_filing(session: requests.Session, ticker: str, cik: str, fy: int, info: dict) -> FilingRecord:
    url = build_document_url(cik, info["accession_no"], info["primary_document"])
    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"FY{fy}.html"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("%s FY%d → skip (exists, %d bytes)", ticker, fy, out_path.stat().st_size)
        body = out_path.read_bytes()
    else:
        log.info("%s FY%d → %s", ticker, fy, url)
        resp = polite_get(session, url)
        body = resp.content
        out_path.write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    rec = FilingRecord(
        ticker=ticker,
        cik=cik,
        fiscal_year=fy,
        accession_no=info["accession_no"],
        period_of_report=info["period_of_report"],
        filing_date=info["filing_date"],
        primary_document=info["primary_document"],
        document_url=url,
        local_path=str(out_path.relative_to(REPO_ROOT)),
        size_bytes=len(body),
        sha256=sha,
    )
    return rec


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    records: list[FilingRecord] = []
    failures: list[dict] = []

    expected_count = 0
    for ticker, cik in TICKERS.items():
        fys = TICKER_FYS[ticker]
        expected_count += len(fys)
        try:
            submissions = load_submissions(session, cik)
        except Exception as e:  # noqa: BLE001
            log.error("submissions fetch failed for %s (%s): %s", ticker, cik, e)
            for fy in fys:
                failures.append({"ticker": ticker, "fiscal_year": fy, "stage": "submissions", "error": str(e)})
            continue

        for fy in fys:
            info = pick_10k_for_fy(session, submissions, fy)
            if info is None:
                log.error("no 10-K with period=%d for %s", fy, ticker)
                failures.append({"ticker": ticker, "fiscal_year": fy, "stage": "match", "error": "no filing found"})
                continue
            try:
                rec = download_filing(session, ticker, cik, fy, info)
                records.append(rec)
            except Exception as e:  # noqa: BLE001
                log.error("download failed for %s FY%d: %s", ticker, fy, e)
                failures.append({"ticker": ticker, "fiscal_year": fy, "stage": "download", "error": str(e)})

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tickers": list(TICKERS.keys()),
        "ticker_fiscal_years": {t: list(fys) for t, fys in TICKER_FYS.items()},
        "expected_count": expected_count,
        "downloaded_count": len(records),
        "failure_count": len(failures),
        "filings": [asdict(r) for r in records],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with FAILURES_PATH.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, sort_keys=True) + "\n")

    log.info("done: %d/%d recorded, %d failures", len(records), manifest["expected_count"], len(failures))
    log.info("manifest → %s", MANIFEST_PATH)
    if failures:
        log.warning("failures → %s", FAILURES_PATH)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
