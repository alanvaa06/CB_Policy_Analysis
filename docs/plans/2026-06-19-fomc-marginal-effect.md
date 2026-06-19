# FOMC Statement Stance — Marginal Effect Beyond the Surprise (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer one question rigorously — does the stance of the FOMC post-meeting statement carry predictive information for market rates BEYOND the known Bauer-Swanson monetary-policy surprise? Build a real stance signal from standalone statements (1999+) via FOMC-RoBERTa and test whether adding stance to a surprise-only model improves out-of-sample R² (nested-model comparison).

**Architecture:** Extends the Phase 0 `src/cbp` package. New one-responsibility modules: statement fetch/parse (`data/fomc_statements.py`), surprise loader (`data/mp_surprise.py`), stance scorer (`models/stance_scorer.py`), nested OOS evaluator (`eval/nested.py`). The leak-safe aligner and walk-forward are EXTENDED (not rewritten) to carry an extra control feature. All network and model inference is isolated behind injected dependencies (an HTML getter; a stance classifier) so every unit of logic is testable offline with fakes — no torch/transformers/requests import in the test suite.

**Tech Stack:** Python 3.11+, pandas, numpy, statsmodels, fredapi (Phase 0). Adds: beautifulsoup4 + requests (fetch/parse), openpyxl (BS xlsx), a tested regex sentence splitter. Heavy inference deps (torch, transformers, huggingface_hub) are an OPTIONAL extra `[infer]` — only the live CLI run needs them. Spec: `docs/prd/002-fomc-marginal-effect.md`.

---

## Prerequisites (manual, before Task 1)

- Phase 0 merged to `main` (done) — `pytest` is green at the start of this plan.
- Python 3.11+ available. Run all `pytest` commands from the repo root.
- **For the live CLI run only (Task 15):**
  - `FRED_API_KEY` in env (Phase 0 requirement, unchanged).
  - Bauer-Swanson surprise file downloaded to `data/raw/monetary-policy-surprises-data.xlsx` (SF Fed: https://www.frbsf.org/economic-research/indicators-data/monetary-policy-surprises/ ). **Confirm the orthogonalized-surprise column name and the back-coverage to 1999 against the actual file** (PRD §11): if it is truncated to recent years, substitute the SF Fed USMPD database or the Bauer-Swanson (2023) replication set. The column name is a parameter of `load_surprise`; the live run passes whatever the file actually uses.
  - FOMC-RoBERTa (`gtfintechlab/FOMC-RoBERTa`, ~1.4 GB) downloads once on first call and is cached by `huggingface_hub`. Install the inference extra: `pip install -e ".[dev,infer]"`. **License: CC BY-NC 4.0 — research use only** (consistent with the research-grade track; blocks commercial deployment).
- The FOMC meeting-date list reuses the existing `data/raw/fomc_dates.csv` (1996–2022 release dates). The live run filters it to `year >= 1999` and passes those dates to `fetch_statements`; non-statement dates (e.g. minutes releases) 404 and are skipped, by design.

## Shared interfaces (defined once, reused exactly)

```python
# src/cbp/config.py  (EXTENDED — EFFR dropped, DGS1 added, roberta id added)
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)
    n0: int = 20
    target_series: tuple[str, ...] = ("DGS2", "DGS1")   # EFFR dropped (mechanical)
    event_window: tuple[int, int] = (-1, 1)
    data_dir: Path = Path("data")
    fred_api_key: str | None = None
    roberta_model_id: str = "gtfintechlab/FOMC-RoBERTa"
```

DataFrame contracts (plain `pd.DataFrame`, validated by tests):
- **Statements** — columns `[date: datetime64, text: str]`, one row per FOMC statement.
- **StanceScores** — columns `[date: datetime64, stance: float]`, per-statement mean of mapped sentence labels.
- **StanceSeries** (release-aligned, as Phase 0) — `[release_date, release_ts: datetime64[UTC], stance, doc_type]`.
- **Surprise** — columns `[date: datetime64, surprise: float]`, BS orthogonalized surprise (one per meeting).
- **AlignedPanel** — `[release_ts, stance, surprise, <sid>_h<h> ...]` (targets strictly after `release_ts`; `surprise` joined on the release calendar date).

Function signatures used across tasks (exact names/shapes — do not drift):
- `parse_statement_html(html: str) -> str`
- `statement_urls(d: datetime.date) -> list[str]`
- `fetch_statements(dates: Iterable[datetime.date], cache_dir: Path, get_html: HtmlGetter = ...) -> pd.DataFrame`  → Statements
- `split_sentences(text: str) -> list[str]`
- `score_statements(statements: pd.DataFrame, classifier: StanceClassifier) -> pd.DataFrame`  → StanceScores
- `load_fomc_roberta(model_id: str = ..., max_length: int = 256) -> StanceClassifier`
- `load_surprise(path: Path, sheet_name=0, date_col="date", surprise_col="MP1_orthogonal") -> pd.DataFrame`  → Surprise
- `stance_frame_from_scores(scores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame`  → StanceSeries
- `build_aligned_panel(market, stance, config, extra_features=None) -> pd.DataFrame`  (EXTENDED)
- `run_walkforward(panel, target_col, model, baseline, n0, feature_cols=("stance",)) -> pd.DataFrame`  (EXTENDED)
- `nested_oos(panel, target_col, n0) -> dict`  (keys: `r2_base, r2_full, delta_r2, n, stance_partial_t`)
- `residual_stance_regression(panel, target_col, n0) -> dict`  (keys: `slope, tstat, r2, n`)
- `run_nested_report(market, stance, surprise, config) -> dict`

**Protocols / type aliases:**
```python
from typing import Protocol, Callable, Optional

class StanceClassifier(Protocol):
    def __call__(self, texts: list[str]) -> list[dict]: ...   # HF pipeline shape: [{"label": "LABEL_0", "score": ...}, ...]

HtmlGetter = Callable[[str], Optional[str]]   # url -> html, or None on 404/error
```

**Label mapping (FOMC-RoBERTa):** `LABEL_0 = Dovish = -1.0`, `LABEL_1 = Hawkish = +1.0`, `LABEL_2 = Neutral = 0.0`.

**Two deliberate deviations from the PRD §4 signature sketch (minimal-impact, per project CLAUDE.md "Minimal Impact"):**
1. `fetch_statements` takes `dates` (+ injectable `get_html`) instead of `start_year`. Reason: the FOMC calendar already exists as a loadable file; passing dates decouples fetch from file layout and makes it testable with a fake getter (matches the repo's DI ethos — `FredClient`, injected `classifier`). The CLI does the `year >= 1999` filter at the call site.
2. `run_walkforward` gains `feature_cols` as a **keyword with default `("stance",)`** rather than a new positional 3rd arg. Reason: this preserves all existing Phase 0 call sites (5 walk-forward tests + `cli.run_report`) bit-for-bit — `SimpleOLS` is already multivariate, so only the `X` column selection changes.

---

## File structure

```
NEW:
  src/cbp/data/fomc_statements.py   # statement_urls (pure) + parse_statement_html (pure) + fetch_statements (cache, DI)
  src/cbp/data/mp_surprise.py       # load_surprise (BS orthogonalized xlsx -> [date, surprise])
  src/cbp/models/stance_scorer.py   # split_sentences (pure) + score_statements (DI classifier) + load_fomc_roberta (factory)
  src/cbp/eval/nested.py            # nested_oos + residual_stance_regression
  tests/test_fomc_statements.py
  tests/test_mp_surprise.py
  tests/test_stance_scorer.py
  tests/test_nested.py

EXTENDED:
  src/cbp/config.py                 # targets (DGS2, DGS1), roberta_model_id
  src/cbp/data/stance.py            # + stance_frame_from_scores; load_stance delegates to it
  src/cbp/align/aligner.py          # build_aligned_panel gains extra_features (surprise) join
  src/cbp/eval/walkforward.py       # run_walkforward gains feature_cols=("stance",)
  src/cbp/cli.py                    # run_nested_report + Phase 1 wiring (--mode phase1)
  pyproject.toml                    # + beautifulsoup4, requests, openpyxl; optional [infer] extra
  tests/test_config.py              # targets updated (DGS2,DGS1) + roberta id
  tests/test_walkforward.py         # + multivariate feature_cols test
  tests/test_aligner.py             # + extra_features join tests
  tests/test_stance.py              # + stance_frame_from_scores test
  tests/test_cli.py                 # + run_nested_report offline test
```

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml:6-9`

- [ ] **Step 1: Add the new runtime deps and the optional `infer` extra**

Edit the `dependencies` line and `[project.optional-dependencies]` block:

```toml
dependencies = ["pandas>=2.0", "numpy>=1.26", "scipy>=1.11", "statsmodels>=0.14", "fredapi>=0.5", "pyarrow>=14", "beautifulsoup4>=4.12", "requests>=2.31", "openpyxl>=3.1"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.0"]
infer = ["torch>=2.2", "transformers>=4.40", "huggingface_hub>=0.23"]
```

- [ ] **Step 2: Install the new deps**

Run: `pip install -e ".[dev]"`
Expected: installs `beautifulsoup4`, `requests`, `openpyxl` (and re-resolves existing). The `[infer]` extra is NOT installed here — the test suite never imports torch/transformers.

- [ ] **Step 3: Verify the suite still imports/passes after the dep change**

Run: `pytest -q`
Expected: PASS (same 30+ Phase 0 tests green; nothing changed in code yet).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add bs4/requests/openpyxl deps + optional [infer] extra (Phase 1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Config — targets DGS2/DGS1, RoBERTa id

**Files:**
- Modify: `src/cbp/config.py:9` (and add a field)
- Test: `tests/test_config.py:9` (update assertions)

- [ ] **Step 1: Update the failing test first**

Edit `tests/test_config.py` so `test_defaults` expects the Phase 1 targets and the new field:

```python
def test_defaults():
    c = Config()
    assert c.horizons == (1, 5, 22)
    assert c.n0 == 20
    assert c.target_series == ("DGS2", "DGS1")     # EFFR dropped (mechanical), DGS1 added
    assert c.event_window == (-1, 1)
    assert isinstance(c.data_dir, Path)
    assert c.roberta_model_id == "gtfintechlab/FOMC-RoBERTa"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_defaults -v`
Expected: FAIL — `assert ("DGS2", "EFFR") == ("DGS2", "DGS1")` and/or `AttributeError: ... 'roberta_model_id'`.

- [ ] **Step 3: Update the Config dataclass**

In `src/cbp/config.py`, change the target tuple and add the model id:

```python
# src/cbp/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)
    n0: int = 20
    target_series: tuple[str, ...] = ("DGS2", "DGS1")
    event_window: tuple[int, int] = (-1, 1)
    data_dir: Path = Path("data")
    fred_api_key: str | None = None
    roberta_model_id: str = "gtfintechlab/FOMC-RoBERTa"
```

- [ ] **Step 4: Run config + full suite**

Run: `pytest tests/test_config.py -v`
Expected: PASS.
Run: `pytest -q`
Expected: PASS. (Note: `tests/test_cli.py::test_run_report_offline` and `tests/test_aligner.py` pass an explicit `Config(target_series=("DGS2",))`, so the default change does not affect them.)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/config.py tests/test_config.py
git commit -m "feat: Phase 1 config — targets DGS2/DGS1, drop EFFR, add roberta_model_id

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `parse_statement_html` — HTML → statement text (both eras)

**Files:**
- Create: `src/cbp/data/fomc_statements.py`
- Test: `tests/test_fomc_statements.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fomc_statements.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fomc_statements.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.data.fomc_statements'`.

- [ ] **Step 3: Write the parser**

```python
# src/cbp/data/fomc_statements.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fomc_statements.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/data/fomc_statements.py tests/test_fomc_statements.py
git commit -m "feat: parse_statement_html — FOMC statement body, modern + historical eras

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `statement_urls` — per-date candidate URLs

**Files:**
- Modify: `src/cbp/data/fomc_statements.py`
- Test: `tests/test_fomc_statements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fomc_statements.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fomc_statements.py::test_statement_urls_modern_era -v`
Expected: FAIL — `ImportError: cannot import name 'statement_urls'`.

- [ ] **Step 3: Implement the URL builder**

Add to `src/cbp/data/fomc_statements.py` (above `parse_statement_html` is fine):

```python
import datetime as dt

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fomc_statements.py -v`
Expected: PASS (5 tests now).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/data/fomc_statements.py tests/test_fomc_statements.py
git commit -m "feat: statement_urls — per-date candidate URLs for both fed eras

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `fetch_statements` — cached fetch with injected getter, skip-on-failure

**Files:**
- Modify: `src/cbp/data/fomc_statements.py`
- Test: `tests/test_fomc_statements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fomc_statements.py`:

```python
import logging

def test_fetch_statements_parses_caches_and_skips(tmp_path, caplog):
    # Fake getter: serves modern fixture for 2008-01-30, 404 (None) for everything else.
    calls = []
    def fake_get(url):
        calls.append(url)
        if url == "https://www.federalreserve.gov/newsevents/pressreleases/monetary20080130a.htm":
            return MODERN_HTML
        return None

    dates = [dt.date(2008, 1, 30), dt.date(2008, 3, 18)]  # 2nd date 404s -> skipped
    cache = tmp_path / "statements"
    with caplog.at_level(logging.WARNING, logger="cbp.data.fomc_statements"):
        out = fetch_statements(dates, cache, get_html=fake_get)

    assert list(out.columns) == ["date", "text"]
    assert len(out) == 1                                   # the 404 date is dropped
    assert out["date"].iloc[0] == __import__("pandas").Timestamp("2008-01-30")
    assert "raise the target range" in out["text"].iloc[0]
    assert (cache / "20080130.html").exists()             # raw HTML cached
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "2008-03-18" in msg                            # the skip is logged, not fabricated

def test_fetch_statements_uses_cache_on_second_call(tmp_path):
    def fake_get(url):
        return MODERN_HTML if "20080130a" in url else None
    dates = [dt.date(2008, 1, 30)]
    cache = tmp_path / "statements"
    fetch_statements(dates, cache, get_html=fake_get)     # populates cache

    n_calls = []
    def counting_get(url):
        n_calls.append(url)
        return MODERN_HTML
    out = fetch_statements(dates, cache, get_html=counting_get)
    assert len(out) == 1
    assert n_calls == []                                  # served from cache, getter untouched

from cbp.data.fomc_statements import fetch_statements  # noqa: E402  (import after fixtures)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fomc_statements.py::test_fetch_statements_parses_caches_and_skips -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_statements'`.

- [ ] **Step 3: Implement the fetcher**

Add to `src/cbp/data/fomc_statements.py`:

```python
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional
import pandas as pd

logger = logging.getLogger(__name__)

HtmlGetter = Callable[[str], Optional[str]]

def _default_get_html(url: str) -> Optional[str]:
    import requests
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fomc_statements.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/data/fomc_statements.py tests/test_fomc_statements.py
git commit -m "feat: fetch_statements — cached, DI getter, skip-and-log on 404/empty

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `split_sentences` — deterministic sentence splitter

**Files:**
- Create: `src/cbp/models/stance_scorer.py`
- Test: `tests/test_stance_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stance_scorer.py
from cbp.models.stance_scorer import split_sentences

def test_splits_on_terminal_punctuation():
    text = "The Committee raised rates. Inflation remains elevated! Will it persist?"
    assert split_sentences(text) == [
        "The Committee raised rates.",
        "Inflation remains elevated!",
        "Will it persist?",
    ]

def test_collapses_whitespace_and_newlines():
    text = "First sentence.\nSecond sentence.   Third sentence."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

def test_empty_and_blank_return_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stance_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.models.stance_scorer'`.

- [ ] **Step 3: Implement the splitter**

```python
# src/cbp/models/stance_scorer.py
from __future__ import annotations
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences on terminal punctuation followed by whitespace.

    Deterministic and dependency-free. A crude splitter (abbreviations like "U.S."
    may over-split) — acceptable because stance is a mean over sentences; recorded
    as a caveat (PRD §11). Returns [] for empty/blank input.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stance_scorer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/stance_scorer.py tests/test_stance_scorer.py
git commit -m "feat: split_sentences — deterministic regex sentence splitter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `score_statements` — sentences → hawk/dove/neutral → mean (injected classifier)

**Files:**
- Modify: `src/cbp/models/stance_scorer.py`
- Test: `tests/test_stance_scorer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stance_scorer.py`:

```python
import pandas as pd
from cbp.models.stance_scorer import score_statements, LABEL_MAP

def _fake_classifier(texts: list[str]) -> list[dict]:
    # Deterministic stand-in for FOMC-RoBERTa: keyword -> label.
    out = []
    for t in texts:
        low = t.lower()
        if "hike" in low or "raise" in low:
            out.append({"label": "LABEL_1", "score": 0.9})   # Hawkish -> +1
        elif "cut" in low or "lower" in low:
            out.append({"label": "LABEL_0", "score": 0.9})   # Dovish -> -1
        else:
            out.append({"label": "LABEL_2", "score": 0.9})   # Neutral -> 0
    return out

def test_label_map_values():
    assert LABEL_MAP == {"LABEL_0": -1.0, "LABEL_1": 1.0, "LABEL_2": 0.0}

def test_score_is_mean_of_mapped_labels():
    # 2 hawkish + 1 neutral -> (1 + 1 + 0) / 3 = 0.6666...
    stmts = pd.DataFrame({
        "date": [pd.Timestamp("2008-01-30")],
        "text": ["We raise the rate. We will hike further. We monitor data."],
    })
    out = score_statements(stmts, _fake_classifier)
    assert list(out.columns) == ["date", "stance"]
    assert out["stance"].iloc[0] == pytest.approx(2.0 / 3.0)

def test_dovish_statement_scores_negative():
    stmts = pd.DataFrame({
        "date": [pd.Timestamp("2008-09-16")],
        "text": ["We cut the rate. We will lower further."],   # -1, -1 -> -1.0
    })
    out = score_statements(stmts, _fake_classifier)
    assert out["stance"].iloc[0] == pytest.approx(-1.0)

import pytest  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stance_scorer.py::test_score_is_mean_of_mapped_labels -v`
Expected: FAIL — `ImportError: cannot import name 'score_statements'`.

- [ ] **Step 3: Implement scorer + label map + Protocol**

Add to `src/cbp/models/stance_scorer.py`:

```python
import logging
from typing import Protocol
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, float] = {"LABEL_0": -1.0, "LABEL_1": 1.0, "LABEL_2": 0.0}  # Dovish/Hawkish/Neutral

class StanceClassifier(Protocol):
    def __call__(self, texts: list[str]) -> list[dict]: ...

def score_statements(statements: pd.DataFrame, classifier: StanceClassifier) -> pd.DataFrame:
    """Score each statement's stance = mean of mapped per-sentence labels.

    Splits each statement into sentences, classifies each with the injected
    `classifier` (HF-pipeline shape: [{"label": "LABEL_x", ...}, ...]), maps
    labels via LABEL_MAP, and averages. Statements with no sentences are skipped
    and logged. Returns columns [date, stance].
    """
    rows = []
    for _, r in statements.iterrows():
        sentences = split_sentences(r["text"])
        if not sentences:
            logger.warning("No sentences for statement %s; skipping", r["date"])
            continue
        preds = classifier(sentences)
        mapped = [LABEL_MAP[p["label"]] for p in preds]
        rows.append({"date": r["date"], "stance": float(np.mean(mapped))})
    return pd.DataFrame(rows, columns=["date", "stance"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stance_scorer.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/stance_scorer.py tests/test_stance_scorer.py
git commit -m "feat: score_statements — per-sentence stance mean via injected classifier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `load_fomc_roberta` — lazy factory wiring the real pipeline

**Files:**
- Modify: `src/cbp/models/stance_scorer.py`
- Test: `tests/test_stance_scorer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stance_scorer.py`:

```python
import sys, types
from cbp.models.stance_scorer import load_fomc_roberta

def test_load_fomc_roberta_wires_pipeline_lazily(monkeypatch):
    captured = {}
    fake_transformers = types.ModuleType("transformers")
    def fake_pipeline(task, model, truncation, max_length):
        captured.update(task=task, model=model, truncation=truncation, max_length=max_length)
        return lambda texts: [{"label": "LABEL_2", "score": 1.0} for _ in texts]
    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    classify = load_fomc_roberta()
    out = classify(["sentence one", "sentence two"])

    assert captured["task"] == "text-classification"
    assert captured["model"] == "gtfintechlab/FOMC-RoBERTa"
    assert captured["truncation"] is True
    assert captured["max_length"] == 256
    assert len(out) == 2 and out[0]["label"] == "LABEL_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stance_scorer.py::test_load_fomc_roberta_wires_pipeline_lazily -v`
Expected: FAIL — `ImportError: cannot import name 'load_fomc_roberta'`.

- [ ] **Step 3: Implement the factory**

Add to `src/cbp/models/stance_scorer.py`:

```python
def load_fomc_roberta(model_id: str = "gtfintechlab/FOMC-RoBERTa", max_length: int = 256) -> StanceClassifier:
    """Lazily build the real FOMC-RoBERTa text-classification pipeline.

    `transformers` is imported inside the function so the test suite (which injects
    a fake classifier) never needs torch/transformers. The model (~1.4GB) is
    downloaded once and cached by huggingface_hub. Per-sentence inputs are
    truncated to `max_length` tokens. License: CC BY-NC 4.0 (research use only).
    """
    from transformers import pipeline
    pipe = pipeline("text-classification", model=model_id, truncation=True, max_length=max_length)

    def classify(texts: list[str]) -> list[dict]:
        return pipe(texts)

    return classify
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stance_scorer.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/stance_scorer.py tests/test_stance_scorer.py
git commit -m "feat: load_fomc_roberta — lazy factory for the real classifier pipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `load_surprise` — Bauer-Swanson orthogonalized series

**Files:**
- Create: `src/cbp/data/mp_surprise.py`
- Test: `tests/test_mp_surprise.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mp_surprise.py
import pandas as pd
from cbp.data.mp_surprise import load_surprise

def test_load_surprise_reads_date_and_orthogonal_column(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({
        "date": ["2008-01-30", "2008-03-18", "2008-04-30"],
        "MP1_orthogonal": [0.05, -0.10, 0.02],
        "ignored_col": [9, 9, 9],
    }).to_excel(p, index=False)

    out = load_surprise(p)
    assert list(out.columns) == ["date", "surprise"]
    assert out["surprise"].tolist() == [0.05, -0.10, 0.02]
    assert out["date"].dt.tz is None                         # tz-naive calendar dates
    assert out["date"].iloc[0] == pd.Timestamp("2008-01-30")

def test_load_surprise_drops_missing_and_sorts(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({
        "date": ["2008-03-18", "2008-01-30"],                # out of order
        "MP1_orthogonal": [None, 0.05],                      # one missing -> dropped
    }).to_excel(p, index=False)

    out = load_surprise(p)
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2008-01-30")
    assert out["surprise"].iloc[0] == 0.05

def test_load_surprise_custom_column_names(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({"meeting": ["2010-11-03"], "u_t": [0.123]}).to_excel(p, index=False)
    out = load_surprise(p, date_col="meeting", surprise_col="u_t")
    assert out["surprise"].iloc[0] == 0.123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mp_surprise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.data.mp_surprise'`.

- [ ] **Step 3: Implement the loader**

```python
# src/cbp/data/mp_surprise.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_surprise(
    path: Path,
    sheet_name: str | int = 0,
    date_col: str = "date",
    surprise_col: str = "MP1_orthogonal",
) -> pd.DataFrame:
    """Load the Bauer-Swanson orthogonalized monetary-policy-surprise series.

    Returns columns [date: datetime64 (tz-naive, normalized), surprise: float],
    one row per meeting, dropping rows with missing surprise, sorted by date.
    Column names are parameters: confirm them against the actual SF Fed file
    (PRD §11) and pass overrides if they differ from the defaults.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]).dt.normalize(),
        "surprise": pd.to_numeric(raw[surprise_col], errors="coerce"),
    })
    return out.dropna(subset=["surprise"]).sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mp_surprise.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/data/mp_surprise.py tests/test_mp_surprise.py
git commit -m "feat: load_surprise — Bauer-Swanson orthogonalized surprise (xlsx)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `stance_frame_from_scores` — attach release timestamps (DRY with load_stance)

**Files:**
- Modify: `src/cbp/data/stance.py`
- Test: `tests/test_stance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stance.py`:

```python
from cbp.data.stance import stance_frame_from_scores

def test_stance_frame_from_scores_joins_calendar(tmp_path):
    cal_p = tmp_path / "fomc_dates.csv"
    cal_p.write_text("release_date\n2020-01-29\n2020-03-18\n")
    cal = load_fomc_calendar(cal_p)
    scores = pd.DataFrame({
        "date": [pd.Timestamp("2020-03-18"), pd.Timestamp("2020-01-29")],  # out of order
        "stance": [-0.8, 0.5],
    })
    out = stance_frame_from_scores(scores, cal)
    assert list(out.columns) == ["release_date", "release_ts", "stance", "doc_type"]
    assert out["stance"].tolist() == [0.5, -0.8]            # sorted by release_ts
    assert out["release_ts"].dt.tz is not None
    assert (out["doc_type"] == "statement").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stance.py::test_stance_frame_from_scores_joins_calendar -v`
Expected: FAIL — `ImportError: cannot import name 'stance_frame_from_scores'`.

- [ ] **Step 3: Add the helper and delegate `load_stance` to it**

Rewrite `src/cbp/data/stance.py`:

```python
# src/cbp/data/stance.py
from pathlib import Path
import pandas as pd

def stance_frame_from_scores(scores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Join per-statement stance scores [date, stance] onto the FOMC calendar to
    produce the release-aligned StanceSeries [release_date, release_ts, stance,
    doc_type]. Single source of truth for the calendar join (used by both the
    Phase 1 scored path and the Phase 0 CSV path).
    """
    s = scores.copy()
    s["release_date"] = pd.to_datetime(s["date"])
    merged = s.merge(calendar, on="release_date", how="inner")
    merged["doc_type"] = "statement"
    return merged[["release_date", "release_ts", "stance", "doc_type"]].sort_values(
        "release_ts"
    ).reset_index(drop=True)

def load_stance(path: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    scores = pd.DataFrame({"date": raw["date"], "stance": raw["stance"]})
    return stance_frame_from_scores(scores, calendar)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stance.py -v`
Expected: PASS (2 tests — the existing `test_stance_joins_release_ts` still passes; behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/data/stance.py tests/test_stance.py
git commit -m "refactor: extract stance_frame_from_scores; load_stance delegates (DRY)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Aligner — join `extra_features` (surprise), preserve no-leak invariant

**Files:**
- Modify: `src/cbp/align/aligner.py:26-57` (`build_aligned_panel`)
- Test: `tests/test_aligner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_aligner.py`:

```python
def test_panel_joins_extra_features_on_release_date():
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    surprise = pd.DataFrame({"date": pd.to_datetime(["2020-01-29"]), "surprise": [0.07]})
    cfg = Config(horizons=(1,), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg, extra_features=surprise)
    assert "surprise" in panel.columns
    assert panel["surprise"].iloc[0] == 0.07
    assert "stance" in panel.columns
    # no-leak invariant unchanged: target still reads exactly one future bar.
    assert abs(panel["DGS2_h1"].iloc[0] - 0.01) < 1e-9

def test_panel_drops_release_missing_extra_feature(caplog):
    import logging
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    surprise = pd.DataFrame({"date": pd.to_datetime(["2019-12-11"]), "surprise": [0.07]})  # different date
    cfg = Config(horizons=(1,), target_series=("DGS2",))
    with caplog.at_level(logging.WARNING, logger="cbp.align.aligner"):
        panel = build_aligned_panel(m, stance, cfg, extra_features=surprise)
    assert panel.empty                                       # release has no surprise -> dropped
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "2020-01-29" in msg and "extra feature" in msg.lower()

def test_panel_without_extra_features_unchanged():
    # Backward compat: omitting extra_features must produce the Phase 0 panel exactly.
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert "surprise" not in panel.columns
    assert abs(panel["DGS2_h1"].iloc[0] - 0.01) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aligner.py::test_panel_joins_extra_features_on_release_date -v`
Expected: FAIL — `TypeError: build_aligned_panel() got an unexpected keyword argument 'extra_features'`.

- [ ] **Step 3: Extend `build_aligned_panel`**

Replace `build_aligned_panel` in `src/cbp/align/aligner.py` with:

```python
def build_aligned_panel(market: pd.DataFrame, stance: pd.DataFrame, config: Config, extra_features: pd.DataFrame | None = None) -> pd.DataFrame:
    # release_ts is tz-aware UTC; real FRED data arrives tz-naive. Normalize a
    # tz-naive market index to UTC so the index/ts comparisons in forward_change
    # are valid (a naive calendar date is treated as that date at 00:00 UTC).
    if isinstance(market.index, pd.DatetimeIndex) and market.index.tz is None:
        market = market.copy()
        market.index = market.index.tz_localize("UTC")

    # Optional control features (e.g. BS surprise) joined on the release CALENDAR
    # date. Indexed by normalized date for O(1) lookup; one row per meeting.
    feat_lookup = None
    feat_cols: list[str] = []
    if extra_features is not None:
        ef = extra_features.copy()
        ef["date"] = pd.to_datetime(ef["date"]).dt.normalize()
        feat_cols = [c for c in ef.columns if c != "date"]
        feat_lookup = ef.set_index("date")

    rows = []
    for _, r in stance.sort_values("release_ts").iterrows():
        row = {"release_ts": r["release_ts"], "stance": r["stance"]}
        ok = True
        reasons: list[str] = []
        if feat_lookup is not None:
            key = pd.to_datetime(r["release_date"]).normalize()
            if key not in feat_lookup.index:
                ok = False
                reasons.append(f"missing extra feature(s) for release date {key.date()}")
            else:
                frow = feat_lookup.loc[key]
                for c in feat_cols:
                    row[c] = float(frow[c])
        for sid in config.target_series:
            if sid not in market.columns:
                ok = False
                reasons.append(f"series {sid} absent from market frame")
                break
            for h in config.horizons:
                val = forward_change(market[sid], r["release_ts"], h)
                if np.isnan(val):
                    ok = False
                    reasons.append(f"({sid}, h={h}) target window incomplete")
                row[f"{sid}_h{h}"] = val
        if ok:
            rows.append(row)
        else:
            logger.warning(
                "Dropping release %s: %s",
                r["release_ts"],
                "; ".join(reasons),
            )
    return pd.DataFrame(rows)
```

Note: when `extra_features` is given, each stance row must carry `release_date` (the StanceSeries always does). The lookup assumes one surprise row per date (BS series is one-per-meeting).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aligner.py -v`
Expected: PASS (all prior aligner tests + 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/align/aligner.py tests/test_aligner.py
git commit -m "feat: aligner extra_features — join surprise on release date, leak-safe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Walk-forward — multivariate `feature_cols`

**Files:**
- Modify: `src/cbp/eval/walkforward.py:10,22`
- Test: `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walkforward.py`:

```python
def test_multivariate_recovers_two_feature_signal():
    rng = np.random.default_rng(3)
    n = 120
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    surprise = rng.normal(size=n)
    stance = rng.normal(size=n)
    target = 1.5 * surprise + 0.8 * stance + 0.05 * rng.normal(size=n)
    p = pd.DataFrame({"release_ts": ts, "surprise": surprise, "stance": stance, "DGS2_h1": target})
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20, feature_cols=["surprise", "stance"])
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert corr > 0.95                                       # both features used -> strong OOS skill

def test_default_feature_cols_is_stance_only():
    # Backward compat: omitting feature_cols selects ["stance"] exactly as Phase 0.
    p = _panel(30)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    assert len(out) == 10
    assert list(out.columns) == ["release_ts", "y_true", "y_pred", "y_base"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_walkforward.py::test_multivariate_recovers_two_feature_signal -v`
Expected: FAIL — `TypeError: run_walkforward() got an unexpected keyword argument 'feature_cols'`.

- [ ] **Step 3: Add `feature_cols` (keyword, default preserves Phase 0)**

In `src/cbp/eval/walkforward.py`, change the signature and the `X` selection:

```python
def run_walkforward(panel: pd.DataFrame, target_col: str, model, baseline, n0: int, feature_cols=("stance",)) -> pd.DataFrame:
    """Expanding-window OOS. For each release i >= n0, train on rows [0, i) and
    predict row i. Training never sees row i or any later row -> no look-ahead.

    `feature_cols` selects the design-matrix columns (default ["stance"] for
    Phase 0 compatibility; Phase 1 nested comparison passes ["surprise"] and
    ["surprise", "stance"]).
    """
    df = panel.sort_values("release_ts").reset_index(drop=True)
    n_skipped = min(n0, len(df))
    logger.info(
        "Skipping first %d leading release(s) below the n0=%d minimum train size",
        n_skipped,
        n0,
    )
    y = df[target_col].to_numpy(dtype=float)
    X = df[list(feature_cols)].to_numpy(dtype=float)
    recs = []
    for i in range(n0, len(df)):
        Xtr, ytr = X[:i], y[:i]
        model.fit(Xtr, ytr); baseline.fit(Xtr, ytr)
        recs.append({
            "release_ts": df["release_ts"].iloc[i],
            "y_true": y[i],
            "y_pred": float(model.predict(X[i:i+1])[0]),
            "y_base": float(baseline.predict(X[i:i+1])[0]),
        })
    return pd.DataFrame(recs, columns=["release_ts", "y_true", "y_pred", "y_base"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_walkforward.py -v`
Expected: PASS (all prior walk-forward tests + 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/eval/walkforward.py tests/test_walkforward.py
git commit -m "feat: run_walkforward feature_cols — multivariate, default stance-only

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: `nested_oos` — surprise-only vs surprise+stance (the answer)

**Files:**
- Create: `src/cbp/eval/nested.py`
- Test: `tests/test_nested.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nested.py
import numpy as np
import pandas as pd
from cbp.eval.nested import nested_oos

def _nested_panel(n, add_stance_signal: bool, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    surprise = rng.normal(size=n)
    stance = rng.normal(size=n)
    coef_stance = 2.0 if add_stance_signal else 0.0
    target = 1.0 * surprise + coef_stance * stance + 0.05 * rng.normal(size=n)
    return pd.DataFrame({"release_ts": ts, "surprise": surprise, "stance": stance, "DGS2_h1": target})

def test_keys_present():
    out = nested_oos(_nested_panel(150, True), "DGS2_h1", n0=20)
    assert set(out) == {"r2_base", "r2_full", "delta_r2", "n", "stance_partial_t"}
    assert out["n"] == 130                                   # 150 - 20 OOS rows

def test_stance_adds_value_delta_r2_positive():
    out = nested_oos(_nested_panel(150, add_stance_signal=True), "DGS2_h1", n0=20)
    assert out["delta_r2"] > 0.05                            # full model beats surprise-only OOS
    assert out["stance_partial_t"] > 3.0                    # in-sample stance coef clearly nonzero

def test_null_stance_adds_nothing_delta_r2_near_zero():
    out = nested_oos(_nested_panel(150, add_stance_signal=False), "DGS2_h1", n0=20)
    assert out["delta_r2"] < 0.02                            # no improvement (small/<=0)
    assert abs(out["stance_partial_t"]) < 2.5                # stance coef not significant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nested.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.eval.nested'`.

- [ ] **Step 3: Implement `nested_oos`**

```python
# src/cbp/eval/nested.py
from __future__ import annotations
import pandas as pd
import statsmodels.api as sm
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward
from cbp.eval.metrics import oos_r2

def _stance_partial_t(panel: pd.DataFrame, target_col: str) -> float:
    """In-sample OLS target ~ const + surprise + stance over the full panel;
    return the t-stat of the stance coefficient. Descriptive companion to the OOS
    delta_r2 (PRD §7), NOT itself an OOS metric.
    """
    df = panel.dropna(subset=[target_col, "surprise", "stance"])
    X = sm.add_constant(df[["surprise", "stance"]].to_numpy(dtype=float))  # [const, surprise, stance]
    res = sm.OLS(df[target_col].to_numpy(dtype=float), X).fit()
    return float(res.tvalues[2])

def nested_oos(panel: pd.DataFrame, target_col: str, n0: int) -> dict:
    """Nested OOS comparison: surprise-only (A) vs surprise+stance (B).

    Runs the Phase 0 walk-forward twice on the same panel and n0 against the same
    ZeroChange baseline; delta_r2 = oos_r2(B) - oos_r2(A). The statement text
    carries marginal predictive information iff delta_r2 > 0.
    """
    a = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise"])
    b = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise", "stance"])
    r2_base = oos_r2(a["y_true"].to_numpy(), a["y_pred"].to_numpy(), a["y_base"].to_numpy())
    r2_full = oos_r2(b["y_true"].to_numpy(), b["y_pred"].to_numpy(), b["y_base"].to_numpy())
    return {
        "r2_base": r2_base,
        "r2_full": r2_full,
        "delta_r2": r2_full - r2_base,
        "n": int(len(b)),
        "stance_partial_t": _stance_partial_t(panel, target_col),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nested.py -v`
Expected: PASS (3 tests — including the synthetic signal ΔR²>0 and null ΔR²≈0 from PRD §10).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/eval/nested.py tests/test_nested.py
git commit -m "feat: nested_oos — surprise-only vs surprise+stance OOS R2 + partial-t

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: `residual_stance_regression` — event-study cross-check

**Files:**
- Modify: `src/cbp/eval/nested.py`
- Test: `tests/test_nested.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nested.py`:

```python
from cbp.eval.nested import residual_stance_regression

def test_residual_regression_picks_up_stance_after_surprise():
    # target = surprise + 2*stance: after a surprise-only OOS fit, the residual is
    # ~2*stance, so regressing residual on stance recovers a clearly positive slope.
    out = residual_stance_regression(_nested_panel(150, add_stance_signal=True), "DGS2_h1", n0=20)
    assert set(out) == {"slope", "tstat", "r2", "n"}
    assert out["n"] == 130
    assert out["slope"] > 0.5
    assert out["tstat"] > 3.0

def test_residual_regression_null_when_no_stance_signal():
    out = residual_stance_regression(_nested_panel(150, add_stance_signal=False), "DGS2_h1", n0=20)
    assert abs(out["tstat"]) < 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nested.py::test_residual_regression_picks_up_stance_after_surprise -v`
Expected: FAIL — `ImportError: cannot import name 'residual_stance_regression'`.

- [ ] **Step 3: Implement the cross-check**

Add to `src/cbp/eval/nested.py`:

```python
def residual_stance_regression(panel: pd.DataFrame, target_col: str, n0: int) -> dict:
    """Cross-check (PRD §7): regress the surprise-only OOS residual on stance.

    residual = y_true - (surprise-only OOS fit). A hawkish-text -> higher-residual
    -yield read independent of the decision. Returns slope/t/r2/n; degrades to NaN
    if fewer than 2 points or stance is constant.
    """
    a = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise"])
    df = panel.sort_values("release_ts").reset_index(drop=True)
    merged = a.merge(df[["release_ts", "stance"]], on="release_ts", how="left")
    resid = (merged["y_true"] - merged["y_pred"]).to_numpy(dtype=float)
    stance = merged["stance"].to_numpy(dtype=float)
    if len(resid) < 2 or len(set(stance)) < 2:
        return {"slope": float("nan"), "tstat": float("nan"), "r2": float("nan"), "n": int(len(resid))}
    res = sm.OLS(resid, sm.add_constant(stance)).fit()
    return {"slope": float(res.params[1]), "tstat": float(res.tvalues[1]), "r2": float(res.rsquared), "n": int(len(resid))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nested.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/eval/nested.py tests/test_nested.py
git commit -m "feat: residual_stance_regression — OOS-residual event-study cross-check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15: CLI — `run_nested_report` (offline-tested) + Phase 1 live wiring

**Files:**
- Modify: `src/cbp/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
from cbp.cli import run_nested_report

def test_run_nested_report_offline():
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rng = np.random.default_rng(0)
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    stance = rng.normal(size=len(rel_ts))
    surprise = rng.normal(size=len(rel_ts))
    market = pd.DataFrame(index=idx)
    base = np.zeros(len(idx))
    for ts, s, u in zip(rel_ts, stance, surprise):
        base[idx > ts] += 0.02 * s + 0.03 * u              # post-release drift = stance + surprise
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))

    rel_date = rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)
    stance_df = pd.DataFrame({"release_ts": rel_ts, "stance": stance, "doc_type": "statement", "release_date": rel_date})
    surprise_df = pd.DataFrame({"date": rel_date, "surprise": surprise})
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))

    report = run_nested_report(market, stance_df, surprise_df, cfg)
    assert ("DGS2", 1) in report["nested"]
    assert ("DGS2", 1) in report["residual"]
    nested = report["nested"][("DGS2", 1)]
    assert set(nested) == {"r2_base", "r2_full", "delta_r2", "n", "stance_partial_t"}
    assert nested["n"] > 0
    assert np.isfinite(nested["delta_r2"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_run_nested_report_offline -v`
Expected: FAIL — `ImportError: cannot import name 'run_nested_report'`.

- [ ] **Step 3: Add `run_nested_report`, a printer, and Phase 1 `main` wiring**

Add to `src/cbp/cli.py` (keep the existing Phase 0 `run_report`/`_print_report`/`main` intact; extend imports and `main`):

```python
from cbp.eval.nested import nested_oos, residual_stance_regression

def run_nested_report(market: pd.DataFrame, stance: pd.DataFrame, surprise: pd.DataFrame, config: Config) -> dict:
    panel = build_aligned_panel(market, stance, config, extra_features=surprise)
    nested, residual = {}, {}
    for sid in config.target_series:
        for h in config.horizons:
            col = f"{sid}_h{h}"
            if col not in panel.columns or len(panel) <= config.n0:
                continue
            nested[(sid, h)] = nested_oos(panel, col, config.n0)
            residual[(sid, h)] = residual_stance_regression(panel, col, config.n0)
    return {"nested": nested, "residual": residual, "n_releases": len(panel)}

def _print_nested(report: dict) -> None:
    print(f"\n=== Nested OOS (surprise-only vs surprise+stance) | aligned releases: {report['n_releases']} ===")
    print(f"{'target':>8} {'h':>3} {'n':>4} {'R2_base':>9} {'R2_full':>9} {'dR2':>9} {'stance_t':>9}")
    for (sid, h), m in report["nested"].items():
        print(f"{sid:>8} {h:>3} {m['n']:>4} {m['r2_base']:>+9.4f} {m['r2_full']:>+9.4f} "
              f"{m['delta_r2']:>+9.4f} {m['stance_partial_t']:>+9.2f}")
    print("\n=== Residual event-study (residual ~ stance, after surprise) ===")
    for (sid, h), e in report["residual"].items():
        print(f"{sid:>8} h={h:>2}: slope={e['slope']:+.4f}  t={e['tstat']:+.2f}  r2={e['r2']:.3f}  n={e['n']}")
```

Then replace `main()` with a `--mode` switch that keeps Phase 0 runnable and adds Phase 1:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description="FOMC stance eval harness")
    ap.add_argument("--mode", choices=["phase0", "phase1"], default="phase1")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2022-12-31")
    args = ap.parse_args()

    from cbp.data.fred import FredClient
    from cbp.data.fomc_calendar import load_fomc_calendar
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY to run the live report.")
    market = FredClient(cfg.fred_api_key).fetch(list(cfg.target_series), args.start, args.end)
    cal = load_fomc_calendar(cfg.data_dir / "raw" / "fomc_dates.csv")

    if args.mode == "phase0":
        from cbp.data.stance import load_stance
        stance = load_stance(cfg.data_dir / "raw" / "tdw_stance.csv", cal)
        _print_report(run_report(market, stance, cfg))
        return

    # phase1: real statements -> FOMC-RoBERTa stance -> BS surprise control -> nested OOS
    from cbp.data.fomc_statements import fetch_statements
    from cbp.models.stance_scorer import score_statements, load_fomc_roberta
    from cbp.data.stance import stance_frame_from_scores
    from cbp.data.mp_surprise import load_surprise

    dates = [d.date() for d in cal["release_date"] if d.year >= 1999]
    statements = fetch_statements(dates, cfg.data_dir / "raw" / "statements")
    scores = score_statements(statements, load_fomc_roberta(cfg.roberta_model_id))
    stance = stance_frame_from_scores(scores, cal)
    surprise = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx")
    _print_nested(run_nested_report(market, stance, surprise, cfg))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (existing `test_run_report_offline` + new `test_run_nested_report_offline`).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/cli.py tests/test_cli.py
git commit -m "feat: CLI Phase 1 — run_nested_report + --mode phase1 live wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 16: Full-suite green + Phase 1 docs

**Files:**
- Modify: `docs/context/todo.md`, `docs/context/results.md`, `docs/context/memory.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: Run the entire suite**

Run: `pytest -q`
Expected: PASS — all Phase 0 tests + every new Phase 1 test green. No torch/transformers/requests import anywhere in the run (fakes only).

- [ ] **Step 2: Record per project CLAUDE.md "Task Management"**

Update the context files (one line each, list format; dedupe before appending):
- `docs/context/results.md`: a 1–4 line review of Phase 1 build (modules added, nested OOS in place, suite green).
- `docs/context/memory.md`: `# decision: Phase 1 isolates statement text's marginal effect via nested OOS (surprise control = BS orthogonalized); EFFR dropped, targets DGS2/DGS1.`
- `docs/context/todo.md`: move the Phase 1 scoping item to done; add `[pending] live Phase 1 run + documented verdict (DoD §3)`.
- `docs/context/sesion-log.md`: `[2026-06-19]: Phase 1 harness implemented (statements + RoBERTa + BS surprise + nested OOS), unit-tested offline.`

If any context file trips `[context-size] OVER CAP`, run `/compact-context` (do NOT inline-compact here).

- [ ] **Step 3: Commit**

```bash
git add docs/context
git commit -m "docs: Phase 1 harness build log + decision + todo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 17 (manual, gated on data): live run + verdict — DoD §2–§3

> Not a unit task — this is the live execution that produces the research answer. Run only after the prerequisites' BS file + `[infer]` extra are in place.

- [ ] **Step 1: Confirm the BS file column name + coverage**

Open `data/raw/monetary-policy-surprises-data.xlsx`; verify the orthogonalized-surprise column name and that it reaches back to 1999. If the default `MP1_orthogonal` is wrong, note the actual name; if coverage is truncated, switch to the SF Fed USMPD / Bauer-Swanson (2023) replication set (PRD §11). If the column differs, pass it through at the `load_surprise` call site in `cli.py:main` (e.g. `load_surprise(path, surprise_col="<actual>")`).

- [ ] **Step 2: Run end-to-end**

Run: `FRED_API_KEY=... python -m cbp.cli --mode phase1 --start 1999-01-01 --end 2022-12-31`
Expected: prints the nested OOS table (ΔR² per target × horizon) + the residual event-study. First run downloads RoBERTa (~1.4GB) and caches statements under `data/raw/statements/`; CPU inference over ~190 statements is minutes.

- [ ] **Step 3: Write the verdict (DoD §3)**

In `docs/results/` (or `docs/context/results.md` if short), record: does statement stance add OOS predictive value beyond the BS surprise on `DGS2`? Apply PRD §7's definition — "text adds value" iff `delta_r2 > 0` on `DGS2` at ≥1 horizon with a consistently-signed stance coefficient; a null everywhere is a valid, publishable result. State the go/no-go for productionizing the text signal, and record the caveats (sentence-mean aggregation; N₀=20 with ~190 statements → noisy magnitudes; CC BY-NC blocks commercial use).

- [ ] **Step 4: Commit**

```bash
git add docs
git commit -m "docs: Phase 1 live verdict — statement stance marginal effect on DGS2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage (PRD §2 in-scope → task):**
- Fetch statements 1999+ (both URL eras), cache, skip-on-fail → Tasks 3–5, 17.
- Score stance with FOMC-RoBERTa (LABEL map, per-statement mean) → Tasks 6–8.
- Load BS orthogonalized surprise control → Task 9.
- Targets DGS2 (primary) + DGS1, EFFR dropped → Task 2.
- Multi-feature aligner + nested OOS (surprise-only vs surprise+stance) → Tasks 11, 13.
- ΔR² per (target, horizon) + residual event-study cross-check → Tasks 13, 14, 15.
- DoD §1 synthetic ΔR²>0 + null ΔR²≈0 → Task 13 tests. DoD §1 multi-feature aligner preserves no-leak → Task 11 tests. DoD §1 HTML→text both eras → Task 3. DoD §1 fake-classifier→mean → Task 7. DoD §1 fixture→surprise → Task 9. DoD §2 CLI end-to-end → Tasks 15, 17. DoD §3 verdict → Task 17.
- Error handling (PRD §8): 404/empty skip → Task 5; missing surprise drop+log → Task 11; window incomplete drop → existing aligner (unchanged); long statements truncated → Task 8 (`truncation=True, max_length`).

**2. Placeholder scan:** No TBD/“add error handling”/“similar to Task N”. Every code step shows complete code; every test step shows the actual test; every run step gives the exact command + expected outcome.

**3. Type/name consistency (checked across tasks):**
- `feature_cols` default `("stance",)` consistent in Task 12 def, Task 13 calls (`["surprise"]`, `["surprise","stance"]`), Task 14 call (`["surprise"]`).
- `nested_oos` returns exactly `{r2_base, r2_full, delta_r2, n, stance_partial_t}` (Task 13 def, Task 13 test, Task 15 printer/test all agree).
- `residual_stance_regression` returns `{slope, tstat, r2, n}` (Task 14 def/test, Task 15 printer all agree).
- `extra_features` join key = release `date` column; StanceSeries carries `release_date` (Task 10) which Task 11 reads — consistent.
- `LABEL_MAP` (`LABEL_0=-1, LABEL_1=+1, LABEL_2=0`) consistent across PRD §6, Task 7 def/test, Task 8 fake.
- `StanceClassifier` Protocol shape (`list[str] -> list[dict]`) consistent across Task 7, Task 8, and the fakes in tests.
- `build_aligned_panel(..., extra_features=None)` — backward-compatible default verified by Task 11's `test_panel_without_extra_features_unchanged` and the untouched Phase 0 aligner tests.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-06-19-fomc-marginal-effect.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best fit here: 16 mechanical TDD tasks with tight, independently-verifiable contracts.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
