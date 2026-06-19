# src/cbp/data/fomc_statements.py
"""FOMC statement HTML parser — handles modern (2006+) and historical (1999-2005) page layouts."""
from __future__ import annotations

from bs4 import BeautifulSoup


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
