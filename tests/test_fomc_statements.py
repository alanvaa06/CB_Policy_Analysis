from cbp.data.fomc_statements import parse_statement_html

# Modern era (2006+): statement body wrapped in <div id="article">; nav <p> sits OUTSIDE it.
MODERN_HTML = """
<html><body>
  <div id="navbar"><p>Skip to main content</p></div>
  <div id="article">
    <p>The Committee decided to raise the target range for the federal funds rate.</p>
    <p>The Committee will continue to monitor incoming information.</p>
  </div>
  <div id="footer"><p>Last update: 2008</p></div>
</body></html>
"""

# Historical era (1999-2005): paragraphs directly in the body, no #article wrapper.
HISTORICAL_HTML = """
<html><body>
  <p>The Federal Open Market Committee decided today to lower its target.</p>
  <p>The Committee believes risks are weighted toward weakness.</p>
</body></html>
"""

def test_parse_modern_extracts_body_excludes_nav():
    text = parse_statement_html(MODERN_HTML)
    assert "raise the target range" in text
    assert "continue to monitor" in text
    assert "Skip to main content" not in text   # nav excluded
    assert "Last update" not in text            # footer excluded

def test_parse_historical_extracts_paragraphs():
    text = parse_statement_html(HISTORICAL_HTML)
    assert "lower its target" in text
    assert "weighted toward weakness" in text

def test_parse_empty_returns_empty_string():
    assert parse_statement_html("<html><body></body></html>") == ""


import datetime as dt
from cbp.data.fomc_statements import statement_urls

def test_statement_urls_modern_era():
    urls = statement_urls(dt.date(2008, 1, 30))
    assert urls == [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20080130a.htm"
    ]

def test_statement_urls_historical_era_has_two_candidates():
    urls = statement_urls(dt.date(2001, 1, 3))
    assert urls == [
        "https://www.federalreserve.gov/boarddocs/press/monetary/2001/20010103/default.htm",
        "https://www.federalreserve.gov/boarddocs/press/general/2001/20010103/default.htm",
    ]
