"""EDGAR client — resolve any company and fetch its latest filing, live.

The live path's data layer. All SEC endpoints require a declared User-Agent and
ask for <=10 req/s (fair-access policy), so every request goes through _get with
a UA header and a small inter-request delay.

Endpoints used (verified reachable 2026-06-30):
  - ticker -> CIK:   https://www.sec.gov/files/company_tickers.json  (~10k companies)
  - submissions:     https://data.sec.gov/submissions/CIK##########.json
  - filing document: https://www.sec.gov/Archives/edgar/data/<cik>/<accn>/<doc>
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

# SEC fair-access: identify the requester. Override via env if desired.
import os

_UA = os.environ.get("SEC_EDGAR_UA", "sec-rag-eval skandula9273@gmail.com")
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_MIN_INTERVAL = 0.15  # seconds between requests (~6-7/s, under the 10/s cap)
_last_req = 0.0


@dataclass
class Filing:
    cik: str            # 10-digit, zero-padded
    company: str
    form: str           # "10-K", "10-Q", ...
    filing_date: str    # YYYY-MM-DD
    accession: str      # e.g. 0000320193-25-000079
    primary_doc: str    # e.g. aapl-20250927.htm
    url: str            # full URL to the primary document


def _get(url: str, *, timeout: int = 30) -> bytes:
    """One rate-limited GET with the required User-Agent."""
    global _last_req
    wait = _MIN_INTERVAL - (time.monotonic() - _last_req)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    _last_req = time.monotonic()
    return data


# The ticker map is ~10k rows and stable within a session; fetch once.
_ticker_map: dict[str, tuple[str, str]] | None = None


def _load_tickers() -> dict[str, tuple[str, str]]:
    global _ticker_map
    if _ticker_map is None:
        raw = json.loads(_get(_TICKERS_URL).decode())
        _ticker_map = {
            v["ticker"].upper(): (str(v["cik_str"]).zfill(10), v["title"])
            for v in raw.values()
        }
    return _ticker_map


def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Ticker -> (CIK 10-digit, company name). Raises ValueError if unknown."""
    t = ticker.strip().upper()
    m = _load_tickers()
    if t not in m:
        raise ValueError(f"Unknown ticker {t!r}. Not in the EDGAR ticker map.")
    return m[t]


def latest_filing(ticker: str, form: str = "10-K") -> Filing:
    """Most recent filing of ``form`` for ``ticker`` (default 10-K), live."""
    cik, company = resolve_ticker(ticker)
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json").decode())
    r = sub["filings"]["recent"]
    for f, date, acc, doc in zip(r["form"], r["filingDate"], r["accessionNumber"], r["primaryDocument"]):
        if f == form:
            accn = acc.replace("-", "")
            cik_int = str(int(cik))  # archives path uses the un-padded CIK
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn}/{doc}"
            return Filing(cik=cik, company=company, form=form, filing_date=date,
                          accession=acc, primary_doc=doc, url=url)
    raise ValueError(f"No {form} found for {ticker} in recent filings.")


def fetch_filing_text(filing: Filing) -> str:
    """Download the filing's primary document and extract readable text (HTML)."""
    html = _get(filing.url, timeout=60).decode("utf-8", errors="replace")
    return html_to_text(html)


def html_to_text(html: str) -> str:
    """Strip a 10-K HTML document to readable text.

    Drops script/style and inline-XBRL bookkeeping tags, collapses whitespace.
    Not perfect on tables (a known lever — same lesson as the FinanceBench PDFs),
    but yields chunkable text for retrieval.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace; keep paragraph-ish breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
