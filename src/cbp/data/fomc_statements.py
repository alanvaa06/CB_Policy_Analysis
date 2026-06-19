# src/cbp/data/fomc_statements.py
"""FOMC statement HTML parser — handles modern (2006+) and historical (1999-2005) page layouts."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HtmlGetter = Callable[[str], Optional[str]]


def statement_urls(d: dt.date) -> list[str]:
    """Candidate URLs for the post-meeting statement on FOMC date `d`, in try order.

    Modern (2006+): /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
    Historical (1999-2005): /boarddocs/press/{monetary|general}/{YYYY}/{YYYYMMDD}/default.htm
    """
    ymd = d.strftime("%Y%m%d")
    if d.year >= 2006:
        return [f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm"]
    return [
        f"https://www.federalreserve.gov/boarddocs/press/monetary/{d.year}/{ymd}/default.htm",
        f"https://www.federalreserve.gov/boarddocs/press/general/{d.year}/{ymd}/default.htm",
    ]


def parse_statement_html(html: str) -> str:
    """Extract the FOMC statement body text from a federalreserve.gov page.

    Modern pages (2006+) wrap the statement in <div id="article">; historical
    pages (1999-2005) put paragraphs directly in the body. Scripts/styles are
    dropped; the #article container (when present) excludes nav/footer noise.
    Returns "" when no body text is found (caller skips on empty).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    container = soup.find(id="article") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    paras = [p for p in paras if p]
    if paras:
        return "\n".join(paras)
    return container.get_text(" ", strip=True)


def _default_get_html(url: str) -> Optional[str]:
    import requests  # lazy import — keeps test suite free of a hard requests dependency
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "cbp-research/0.1"})
        return resp.text if resp.status_code == 200 else None
    except requests.RequestException:
        return None


def fetch_statements(
    dates: Iterable[dt.date],
    cache_dir: Path,
    get_html: HtmlGetter = _default_get_html,
) -> pd.DataFrame:
    """Fetch + parse FOMC statements for `dates`, caching raw HTML under cache_dir.

    Tries each candidate URL (statement_urls) until one returns HTML; caches the
    raw HTML; parses the body. A date whose candidates all 404, or that parses to
    empty text, is LOGGED and SKIPPED (never fabricated). Returns one row per
    successfully fetched statement: columns [date: Timestamp, text: str].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(set(dates)):
        cache_path = cache_dir / f"{d.strftime('%Y%m%d')}.html"
        html: Optional[str] = None
        if cache_path.exists():
            html = cache_path.read_text(encoding="utf-8")
        else:
            for url in statement_urls(d):
                html = get_html(url)
                if html:
                    cache_path.write_text(html, encoding="utf-8")
                    break
        if not html:
            logger.warning("No statement HTML for %s (all URL candidates failed); skipping", d)
            continue
        text = parse_statement_html(html)
        if not text.strip():
            logger.warning("Empty parse for %s; skipping", d)
            continue
        rows.append({"date": pd.Timestamp(d), "text": text})
    return pd.DataFrame(rows, columns=["date", "text"])
