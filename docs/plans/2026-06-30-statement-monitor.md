# FOMC Statement Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An idempotent `python -m cbp.monitor` engine that scores each new FOMC statement (action + hawk/dove lexicon + RoBERTa), appends to a committed tone history, and renders a static interactive Plotly dashboard (tone levels, meeting-over-meeting deltas, latest-vs-prior redline) published to GitHub Pages via a torch-free CI rebuild.

**Architecture:** New `src/cbp/monitor/` subpackage of small pure modules (`history`, `calendar`, `score`, `contrast`, `site`) plus a `__main__` orchestrator. Reuses the existing `fetch_statements`, `score_statements_lexicon`, `score_statements`/`split_sentences` unchanged. Two **committed** artifacts are the contract between a heavy local scoring run and a light CI render: `data/monitor/tone_history.csv` (drives the charts) and `data/monitor/latest_redline.json` (drives the redline panel — CI has no statement HTML cache). The built `site/index.html` is a derived artifact (gitignored, produced in CI, published to `gh-pages`).

**Tech Stack:** Python 3.11+, pandas, stdlib `json`/`re`/`difflib`/`argparse`, pytest. Plotly (new `[site]` extra) imported only inside `site.py`. torch/transformers (`[infer]`) only for a local scoring run, never in tests or CI.

**Spec:** `docs/prd/004-fomc-statement-monitor.md`

**Planning deviations from spec (intentional, recorded):**
- **Added a 2nd committed artifact** `data/monitor/latest_redline.json` (the computed redline segments). The spec named only the CSV as the contract; CI cannot rebuild the redline because the raw statement HTML cache is gitignored, so the segments must be committed. Deltas are still recomputed in CI from the CSV.
- **Dropped the `--backfill` flag** (YAGNI): the default run processes all `pending_dates`, which on an empty history already builds the full 1999→latest history. First run = backfill.

---

### Task 1: Scaffolding — `[site]` extra, `.gitignore`, config paths, package init

**Files:**
- Modify: `pyproject.toml:8-11`
- Modify: `.gitignore:28-33`
- Modify: `src/cbp/config.py:5-14`
- Create: `src/cbp/monitor/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  — APPEND these tests (keep existing ones)
from pathlib import Path
from cbp.config import Config


def test_config_monitor_paths_defaults():
    c = Config()
    assert c.monitor_dir == Path("data/monitor")
    assert c.history_path == Path("data/monitor/tone_history.csv")
    assert c.calendar_path == Path("data/monitor/fomc_calendar.csv")
    assert c.redline_path == Path("data/monitor/latest_redline.json")
    assert c.lexicon_dir == Path("data/lexicons")
    assert c.statements_dir == Path("data/raw/statements")
    assert c.site_out == Path("site/index.html")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_monitor_paths_defaults -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'monitor_dir'`

- [ ] **Step 3: Add the config fields**

```python
# src/cbp/config.py — add these fields inside the Config dataclass, after lexicon_path
    lexicon_dir: Path = Path("data/lexicons")
    statements_dir: Path = Path("data/raw/statements")
    monitor_dir: Path = Path("data/monitor")
    history_path: Path = Path("data/monitor/tone_history.csv")
    calendar_path: Path = Path("data/monitor/fomc_calendar.csv")
    redline_path: Path = Path("data/monitor/latest_redline.json")
    site_out: Path = Path("site/index.html")
```

- [ ] **Step 4: Create the package init**

```python
# src/cbp/monitor/__init__.py
"""FOMC statement monitor: score each new statement and render the dashboard."""
```

- [ ] **Step 5: Add the `[site]` extra to pyproject**

```toml
# pyproject.toml — under [project.optional-dependencies], add this line after viz = [...]
site = ["plotly>=5.20"]
```

- [ ] **Step 6: Un-ignore committed monitor artifacts; ignore the built site**

```gitignore
# .gitignore — after the "!/data/lexicons/" line (around line 30), add:
# ...and the versioned monitor artifacts (tone history, calendar, latest redline)
!/data/monitor/
```
```gitignore
# .gitignore — at the end of the file, add:

# Built dashboard (derived artifact; CI publishes it to gh-pages)
/site/
```

- [ ] **Step 7: Run tests to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all config tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/cbp/config.py src/cbp/monitor/__init__.py tests/test_config.py
git commit -m "feat(monitor): scaffolding — [site] extra, config paths, gitignore artifacts"
```

---

### Task 2: `history.py` — load / upsert / save the committed tone history

**Files:**
- Create: `src/cbp/monitor/history.py`
- Test: `tests/test_monitor_history.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_history.py
import pandas as pd
from cbp.monitor.history import (
    HISTORY_COLUMNS, load_history, upsert_history, save_history,
)


def _row(date, action=0.0, lex=0.0, rob=0.0, n=5):
    return {"date": pd.Timestamp(date), "action": action, "lexicon_tone": lex,
            "roberta_stance": rob, "n_sentences": n}


def test_load_history_missing_returns_empty_schema(tmp_path):
    h = load_history(tmp_path / "nope.csv")
    assert list(h.columns) == HISTORY_COLUMNS
    assert len(h) == 0


def test_upsert_appends_and_sorts(tmp_path):
    h = pd.DataFrame([_row("2024-03-20", action=-1.0)])
    new = pd.DataFrame([_row("2024-01-31", action=1.0)])
    out = upsert_history(h, new)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-03-20"]


def test_upsert_same_date_overwrites(tmp_path):
    h = pd.DataFrame([_row("2024-03-20", action=-1.0, rob=-0.5)])
    new = pd.DataFrame([_row("2024-03-20", action=-1.0, rob=-0.9)])
    out = upsert_history(h, new)
    assert len(out) == 1
    assert out.loc[0, "roberta_stance"] == -0.9   # new wins


def test_save_then_load_roundtrip_is_idempotent(tmp_path):
    p = tmp_path / "hist.csv"
    h = upsert_history(load_history(p), pd.DataFrame([_row("2024-01-31", action=1.0, rob=0.25)]))
    save_history(h, p)
    reloaded = load_history(p)
    assert reloaded.loc[0, "action"] == 1.0
    assert reloaded.loc[0, "roberta_stance"] == 0.25
    # re-upserting the same row is a no-op on length
    again = upsert_history(reloaded, pd.DataFrame([_row("2024-01-31", action=1.0, rob=0.25)]))
    assert len(again) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.history'`

- [ ] **Step 3: Implement `history.py`**

```python
# src/cbp/monitor/history.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = ["date", "action", "lexicon_tone", "roberta_stance", "n_sentences"]


def load_history(path: Path) -> pd.DataFrame:
    """Load the committed tone history. Missing file -> empty frame with the schema
    (date as datetime64). Always returns exactly HISTORY_COLUMNS, sorted by date."""
    path = Path(path)
    if not path.exists():
        empty = {c: pd.Series(dtype="datetime64[ns]" if c == "date" else "float64")
                 for c in HISTORY_COLUMNS}
        return pd.DataFrame(empty)
    df = pd.read_csv(path, parse_dates=["date"])
    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[HISTORY_COLUMNS].sort_values("date").reset_index(drop=True)


def upsert_history(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append `new` rows; on a duplicate date the new row wins. Result is sorted by
    date, de-duplicated, index reset. Idempotent: re-upserting identical rows is a no-op."""
    combined = pd.concat([history, new[HISTORY_COLUMNS]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return (combined.drop_duplicates("date", keep="last")
                    .sort_values("date").reset_index(drop=True))


def save_history(history: pd.DataFrame, path: Path) -> None:
    """Write the history to CSV with date as YYYY-MM-DD (stable, diff-friendly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = history.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitor_history.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/history.py tests/test_monitor_history.py
git commit -m "feat(monitor): tone_history load/upsert/save (idempotent by date)"
```

---

### Task 3: `calendar.py` — meeting-date list + pending-date discovery

**Files:**
- Create: `src/cbp/monitor/calendar.py`
- Test: `tests/test_monitor_calendar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_calendar.py
import datetime as dt
import pandas as pd
import pytest
from cbp.monitor.calendar import load_calendar, pending_dates


def test_load_calendar_parses_sorted_unique(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date\n2024-03-20\n2024-01-31\n2024-01-31\n")
    assert load_calendar(p) == [dt.date(2024, 1, 31), dt.date(2024, 3, 20)]


def test_load_calendar_missing_raises_valueerror():
    with pytest.raises(ValueError, match="calendar"):
        load_calendar("does/not/exist.csv")


def test_pending_dates_excludes_already_scored():
    cal = [dt.date(2024, 1, 31), dt.date(2024, 3, 20), dt.date(2024, 5, 1)]
    history = pd.DataFrame({"date": pd.to_datetime(["2024-01-31"])})
    assert pending_dates(cal, history) == [dt.date(2024, 3, 20), dt.date(2024, 5, 1)]


def test_pending_dates_empty_history_returns_all():
    cal = [dt.date(2024, 1, 31), dt.date(2024, 3, 20)]
    history = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    assert pending_dates(cal, history) == cal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.calendar'`

- [ ] **Step 3: Implement `calendar.py`**

```python
# src/cbp/monitor/calendar.py
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_calendar(path: Path) -> list[dt.date]:
    """Load scheduled FOMC announcement dates from a CSV with a `date` column.
    Returns sorted unique `datetime.date`s. Missing file -> ValueError naming the path."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"FOMC calendar not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return sorted({d.date() for d in df["date"]})


def pending_dates(calendar: list[dt.date], history: pd.DataFrame) -> list[dt.date]:
    """Calendar dates with no row yet in `history`. Sorted ascending."""
    scored = set()
    if len(history):
        scored = {d.date() for d in pd.to_datetime(history["date"])}
    return sorted(d for d in calendar if d not in scored)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitor_calendar.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/calendar.py tests/test_monitor_calendar.py
git commit -m "feat(monitor): calendar load + pending-date discovery"
```

---

### Task 4: `score.py` — score one statement frame on all three measures

**Files:**
- Create: `src/cbp/monitor/score.py`
- Test: `tests/test_monitor_score.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_score.py
import math
import pandas as pd
from cbp.monitor.score import score_all_measures
from cbp.monitor.history import HISTORY_COLUMNS


def _statements():
    # one clearly hawkish action ("raise"), one dovish-stance statement
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "text": ["The Committee decided to raise the target range. Policy is restrictive.",
                 "The Committee decided to lower rates. Policy is accommodative."],
    })


def fake_roberta(texts):
    # all sentences hawkish (+1) -> roberta_stance == 1.0 for every statement
    return [{"label": "LABEL_1"} for _ in texts]


def test_score_all_measures_columns_and_merge():
    out = score_all_measures(_statements(), lexicon_dir="data/lexicons", roberta=fake_roberta)
    assert list(out.columns) == HISTORY_COLUMNS
    assert len(out) == 2
    # n_sentences counted via split_sentences (2 sentences each)
    assert list(out["n_sentences"]) == [2, 2]
    # roberta injected -> all +1.0
    assert all(v == 1.0 for v in out["roberta_stance"])
    # action lexicon: raise -> +1, lower -> -1
    assert out.loc[out["date"] == pd.Timestamp("2024-01-31"), "action"].iloc[0] == 1.0
    assert out.loc[out["date"] == pd.Timestamp("2024-03-20"), "action"].iloc[0] == -1.0


def test_score_all_measures_no_roberta_is_nan():
    out = score_all_measures(_statements(), lexicon_dir="data/lexicons", roberta=None)
    assert out["roberta_stance"].isna().all()
    # the light measures still populate
    assert not out["action"].isna().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.score'`

- [ ] **Step 3: Implement `score.py`**

```python
# src/cbp/monitor/score.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
from cbp.models.stance_scorer import StanceClassifier, score_statements, split_sentences
from cbp.monitor.history import HISTORY_COLUMNS


def score_all_measures(
    statements: pd.DataFrame,
    *,
    lexicon_dir: Path,
    roberta: StanceClassifier | None = None,
) -> pd.DataFrame:
    """Score each statement on all three descriptive measures and return one row per
    statement with exactly HISTORY_COLUMNS.

    - action       : action_tone.json lexicon (+1 raise / -1 lower / 0 hold)
    - lexicon_tone  : hawk_dove.json lexicon net stance
    - roberta_stance: per-sentence-mean stance via the injected `roberta` classifier;
                      NaN (whole column) when `roberta is None`. Statements RoBERTa skips
                      (no sentences) become NaN via the left-merge.
    - n_sentences   : split_sentences count (independent of RoBERTa availability)
    """
    lexicon_dir = Path(lexicon_dir)
    hawk_a, dove_a = load_lexicon(lexicon_dir / "action_tone.json")
    hawk_l, dove_l = load_lexicon(lexicon_dir / "hawk_dove.json")

    act = score_statements_lexicon(statements, hawk_a, dove_a).rename(columns={"stance": "action"})
    lex = score_statements_lexicon(statements, hawk_l, dove_l).rename(columns={"stance": "lexicon_tone"})
    nsent = pd.DataFrame({
        "date": statements["date"].to_numpy(),
        "n_sentences": [len(split_sentences(t)) for t in statements["text"]],
    })

    out = act.merge(lex, on="date").merge(nsent, on="date")
    if roberta is not None:
        rob = score_statements(statements, roberta).rename(columns={"stance": "roberta_stance"})
        out = out.merge(rob, on="date", how="left")
    else:
        out["roberta_stance"] = float("nan")
    return out[HISTORY_COLUMNS]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitor_score.py -v`
Expected: PASS (2 tests). (Relies on the repo `data/lexicons/action_tone.json` scoring `raise`→+1, `lower`→−1.)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/score.py tests/test_monitor_score.py
git commit -m "feat(monitor): score_all_measures (action+lexicon+injected RoBERTa)"
```

---

### Task 5: `contrast.py` — tone deltas + sentence-level redline

**Files:**
- Create: `src/cbp/monitor/contrast.py`
- Test: `tests/test_monitor_contrast.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_contrast.py
import math
import pandas as pd
from cbp.monitor.contrast import MEASURES, tone_deltas, redline


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "action": [1.0, -1.0],
        "lexicon_tone": [0.5, 0.2],
        "roberta_stance": [0.4, float("nan")],
        "n_sentences": [10, 12],
    })


def test_tone_deltas_basic_and_nan():
    d = tone_deltas(_hist())
    assert d["date_latest"] == "2024-03-20" and d["date_prior"] == "2024-01-31"
    assert d["action"]["delta"] == -2.0
    assert math.isclose(d["lexicon_tone"]["delta"], -0.3, abs_tol=1e-9)
    assert d["roberta_stance"]["delta"] is None   # latest is NaN


def test_tone_deltas_needs_two_rows():
    one = _hist().iloc[:1]
    assert tone_deltas(one) == {}


def test_redline_detects_equal_insert_delete_replace():
    prev = "Rates unchanged. Inflation remains elevated. Risks are balanced."
    curr = "Rates unchanged. Inflation has eased. Risks are balanced. New paragraph."
    segs = redline(prev, curr)
    ops = [s["op"] for s in segs]
    assert "equal" in ops      # "Rates unchanged."
    assert "replace" in ops    # inflation sentence reworded
    assert "insert" in ops     # "New paragraph."
    # a pure deletion case
    segs2 = redline("A sentence. B sentence.", "A sentence.")
    assert any(s["op"] == "delete" and "B sentence" in s["prev"] for s in segs2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_contrast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.contrast'`

- [ ] **Step 3: Implement `contrast.py`**

```python
# src/cbp/monitor/contrast.py
from __future__ import annotations

import difflib

import pandas as pd

from cbp.models.stance_scorer import split_sentences

MEASURES = ["action", "lexicon_tone", "roberta_stance"]


def _num(v):
    return None if pd.isna(v) else float(v)


def tone_deltas(history: pd.DataFrame) -> dict:
    """Latest-vs-prior change per measure, from the last two history rows.

    Returns {} when fewer than two rows. Otherwise:
      {"date_prior": "YYYY-MM-DD", "date_latest": "YYYY-MM-DD",
       <measure>: {"prior": float|None, "latest": float|None, "delta": float|None}}
    `delta` is None when either side is NaN (e.g. a --no-roberta gap)."""
    if len(history) < 2:
        return {}
    prior, latest = history.iloc[-2], history.iloc[-1]
    out = {
        "date_prior": pd.Timestamp(prior["date"]).strftime("%Y-%m-%d"),
        "date_latest": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
    }
    for m in MEASURES:
        p, l = _num(prior[m]), _num(latest[m])
        out[m] = {"prior": p, "latest": l,
                  "delta": (None if p is None or l is None else l - p)}
    return out


def redline(prev_text: str, curr_text: str) -> list[dict]:
    """Sentence-level track-changes diff of two statements.

    Splits both with the shared `split_sentences`, runs difflib over the sentence
    lists, and emits ordered segments {op, prev, curr} with
    op in {equal, insert, delete, replace}. Textual, not semantic (see PRD §11)."""
    a, b = split_sentences(prev_text), split_sentences(curr_text)
    segments: list[dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        prev = " ".join(a[i1:i2])
        curr = " ".join(b[j1:j2])
        if op == "insert":
            prev = ""
        elif op == "delete":
            curr = ""
        segments.append({"op": op, "prev": prev, "curr": curr})
    return segments
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitor_contrast.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/contrast.py tests/test_monitor_contrast.py
git commit -m "feat(monitor): tone deltas + difflib sentence redline"
```

---

### Task 6: `site.py` — Plotly figures, redline HTML, self-contained page

**Files:**
- Create: `src/cbp/monitor/site.py`
- Test: `tests/test_monitor_site.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_site.py
import pandas as pd
import pytest

pytest.importorskip("plotly")  # site rendering needs the [site] extra; skip if absent

from cbp.monitor.site import (
    build_levels_figure, build_deltas_figure, build_redline_html, render_site,
)


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01"]),
        "action": [1.0, -1.0, 0.0],
        "lexicon_tone": [0.5, 0.2, 0.0],
        "roberta_stance": [0.4, 0.1, float("nan")],
        "n_sentences": [10, 12, 9],
    })


def test_levels_figure_has_three_traces():
    fig = build_levels_figure(_hist())
    assert len(fig.data) == 3
    assert {t.name for t in fig.data} == {"action", "lexicon", "RoBERTa"}


def test_deltas_figure_has_three_traces():
    fig = build_deltas_figure(_hist())
    assert len(fig.data) == 3


def test_redline_html_classes_per_op():
    segs = [{"op": "equal", "prev": "x", "curr": "x"},
            {"op": "insert", "prev": "", "curr": "new"},
            {"op": "delete", "prev": "gone", "curr": ""},
            {"op": "replace", "prev": "old", "curr": "fresh"}]
    html = build_redline_html(segs)
    for cls in ("rl-equal", "rl-insert", "rl-delete", "rl-replace"):
        assert cls in html
    assert "&lt;" not in "new"  # sanity: escaping only applied to content


def test_render_site_writes_self_contained_html(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "plotly" in html.lower()              # plotly.js inlined
    assert "descriptive monitor" in html.lower() # honest framing banner
    assert "Rates unchanged." in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_site.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.site'` (or SKIP if plotly not installed — install with `pip install -e ".[site]"`)

- [ ] **Step 3: Implement `site.py`**

```python
# src/cbp/monitor/site.py
from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

VERDICT_URL = "https://github.com/"  # replaced with the repo verdict link in Task 10
_LEVELS = [("action", "action"), ("lexicon_tone", "lexicon"), ("roberta_stance", "RoBERTa")]


def build_levels_figure(history: pd.DataFrame):
    """Tone levels through time: one line per measure, gaps left as gaps."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col], name=name,
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="FOMC statement tone — levels (1999→latest)",
                      yaxis_title="tone", template="plotly_white", height=380)
    return fig


def build_deltas_figure(history: pd.DataFrame):
    """Meeting-over-meeting Δ per measure (the delta-history)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col].diff(), name=f"Δ {name}",
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="FOMC statement tone — meeting-over-meeting change",
                      yaxis_title="Δ tone", template="plotly_white", height=380)
    return fig


def build_redline_html(segments: list[dict]) -> str:
    """Render redline segments as <p> blocks with rl-<op> CSS classes."""
    if not segments:
        return '<p class="rl-empty">Need ≥2 statements to show a redline.</p>'
    rows = []
    for s in segments:
        op = s["op"]
        if op == "equal":
            rows.append(f'<p class="rl-equal">{_html.escape(s["curr"])}</p>')
        elif op == "insert":
            rows.append(f'<p class="rl-insert">+ {_html.escape(s["curr"])}</p>')
        elif op == "delete":
            rows.append(f'<p class="rl-delete">− {_html.escape(s["prev"])}</p>')
        else:  # replace
            rows.append(f'<p class="rl-replace">− {_html.escape(s["prev"])}'
                        f'<br>+ {_html.escape(s["curr"])}</p>')
    return "\n".join(rows)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:+.3f}"


def _deltas_table_html(deltas: dict) -> str:
    if not deltas:
        return '<p class="rl-empty">Need ≥2 statements to compute deltas.</p>'
    rows = []
    for col, name in _LEVELS:
        m = deltas[col]
        rows.append(f"<tr><td>{name}</td><td>{_fmt(m['prior'])}</td>"
                    f"<td>{_fmt(m['latest'])}</td><td><b>{_fmt(m['delta'])}</b></td></tr>")
    return (f"<p>Comparing <b>{deltas['date_latest']}</b> vs <b>{deltas['date_prior']}</b></p>"
            "<table class='deltas'><tr><th>measure</th><th>prior</th><th>latest</th>"
            "<th>Δ</th></tr>" + "".join(rows) + "</table>")


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}
.banner{background:#FFF3CD;border:1px solid #E0C97A;padding:10px 14px;border-radius:6px;font-size:14px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px}
table.deltas{border-collapse:collapse}table.deltas td,table.deltas th{border:1px solid #ddd;padding:4px 10px;text-align:right}
table.deltas td:first-child,table.deltas th:first-child{text-align:left}
.redline p{margin:4px 0;padding:4px 8px;border-radius:4px}
.rl-equal{color:#555}.rl-insert{background:#E6F4EA;color:#137333}
.rl-delete{background:#FCE8E6;color:#A50E0E;text-decoration:line-through}
.rl-replace{background:#FEF7E0;color:#8A6D00}.rl-empty{color:#888;font-style:italic}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FOMC Statement Monitor</title><style>{css}</style></head><body>
<h1>FOMC Statement Monitor</h1>
<div class="banner">{banner}</div>
<h2>Latest vs prior — tone deltas</h2>{deltas_table}
<h2>Latest vs prior — statement redline</h2><div class="redline">{redline}</div>
<h2>Tone levels through time</h2>{levels}
<h2>Tone change through time</h2>{deltas_fig}
</body></html>"""


def render_site(history: pd.DataFrame, deltas: dict, segments: list[dict],
                out_path: Path, *, verdict_url: str = VERDICT_URL) -> None:
    """Assemble the self-contained dashboard and write it to `out_path`."""
    levels = build_levels_figure(history).to_html(full_html=False, include_plotlyjs="inline")
    deltas_fig = build_deltas_figure(history).to_html(full_html=False, include_plotlyjs=False)
    banner = ("This is a <b>descriptive monitor, not a predictive signal</b>. FOMC statement "
              "tone adds no out-of-sample value beyond the policy surprise "
              f'(<a href="{verdict_url}">Phase 1/2a verdicts</a>).')
    page = _PAGE.format(css=_CSS, banner=banner, deltas_table=_deltas_table_html(deltas),
                        redline=build_redline_html(segments), levels=levels, deltas_fig=deltas_fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pip install -e ".[site]" && pytest tests/test_monitor_site.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/site.py tests/test_monitor_site.py
git commit -m "feat(monitor): Plotly levels/deltas figures + redline page render"
```

---

### Task 7: `__main__.py` — orchestrator + CLI

**Files:**
- Create: `src/cbp/monitor/__main__.py`
- Test: `tests/test_monitor_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_main.py
import json
import pandas as pd
import pytest

pytest.importorskip("plotly")

from cbp.config import Config
from cbp.monitor import __main__ as m
from cbp.monitor.history import load_history


# fake statement HTML keyed by the URL's date stamp; returns hawkish/dovish bodies
_HTML = {
    "20240131": "<html><body><p>The Committee decided to raise the target range. "
                "Policy is restrictive.</p></body></html>",
    "20240320": "<html><body><p>The Committee decided to lower the target range. "
                "Policy is accommodative.</p></body></html>",
}


def _fake_get_html(url):
    for ymd, body in _HTML.items():
        if ymd in url:
            return body
    return None


def _fake_roberta(texts):
    return [{"label": "LABEL_2"} for _ in texts]  # neutral 0.0


def _cfg(tmp_path) -> Config:
    return Config(
        history_path=tmp_path / "tone_history.csv",
        calendar_path=tmp_path / "cal.csv",
        redline_path=tmp_path / "latest_redline.json",
        statements_dir=tmp_path / "statements",
        site_out=tmp_path / "site" / "index.html",
        lexicon_dir=__import__("pathlib").Path("data/lexicons"),
    )


def test_run_monitor_end_to_end_builds_history_redline_and_site(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.calendar_path.write_text("date\n2024-01-31\n2024-03-20\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)

    hist = load_history(cfg.history_path)
    assert list(hist["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-03-20"]
    assert hist.loc[hist["date"] == "2024-03-20", "action"].iloc[0] == -1.0

    redline = json.loads(cfg.redline_path.read_text())
    assert redline["date_latest"] == "2024-03-20"
    assert any(s["op"] in {"replace", "insert", "delete"} for s in redline["segments"])

    assert cfg.site_out.exists()
    assert "descriptive monitor" in cfg.site_out.read_text(encoding="utf-8").lower()


def test_rebuild_only_does_not_fetch(tmp_path):
    cfg = _cfg(tmp_path)
    # seed a history + redline; NO calendar, NO get_html -> must not touch the network
    cfg.calendar_path.write_text("date\n2024-01-31\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)
    cfg.site_out.unlink()
    # rebuild with a getter that would explode if called
    def _boom(url):
        raise AssertionError("rebuild-only must not fetch")
    m.run_monitor(cfg, rebuild_only=True, get_html=_boom)
    assert cfg.site_out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_main.py -v`
Expected: FAIL with `AttributeError: module 'cbp.monitor.__main__' has no attribute 'run_monitor'`

- [ ] **Step 3: Implement `__main__.py`**

```python
# src/cbp/monitor/__main__.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from cbp.config import Config
from cbp.data.fomc_statements import fetch_statements
from cbp.models.stance_scorer import StanceClassifier
from cbp.monitor.calendar import load_calendar, pending_dates
from cbp.monitor.contrast import redline, tone_deltas
from cbp.monitor.history import load_history, save_history, upsert_history
from cbp.monitor.score import score_all_measures
from cbp.monitor.site import VERDICT_URL, render_site

logger = logging.getLogger(__name__)


def build_classifier(cfg: Config, use_roberta: bool) -> StanceClassifier | None:
    """Lazily build the real RoBERTa pipeline (needs [infer]); None when disabled."""
    if not use_roberta:
        return None
    from cbp.models.stance_scorer import load_fomc_roberta
    return load_fomc_roberta(cfg.roberta_model_id)


def _write_latest_redline(cfg: Config, history: pd.DataFrame) -> None:
    """Persist the redline of the two most recent statements so CI (which lacks the
    HTML cache) can render the panel. Reads both texts from the local cache."""
    if len(history) < 2:
        return
    last2 = [pd.Timestamp(d).date() for d in history["date"].iloc[-2:]]
    texts = fetch_statements(last2, cfg.statements_dir)  # cache hit; no network locally
    tmap = {pd.Timestamp(r.date).date(): r.text for r in texts.itertuples()}
    if last2[0] in tmap and last2[1] in tmap:
        payload = {"date_prior": str(last2[0]), "date_latest": str(last2[1]),
                   "segments": redline(tmap[last2[0]], tmap[last2[1]])}
        Path(cfg.redline_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.redline_path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _load_segments(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("segments", [])


def run_monitor(
    cfg: Config,
    *,
    dates: list[dt.date] | None = None,
    use_roberta: bool = True,
    rebuild_only: bool = False,
    get_html=None,
    roberta: StanceClassifier | None = None,
) -> None:
    """Discover pending statements, score + upsert them, then render the dashboard.

    `get_html`/`roberta` are injection seams for tests; production passes neither and
    the real HTTP fetch + RoBERTa pipeline are used. `rebuild_only` skips all fetching
    and scoring and just re-renders from the committed CSV + redline JSON (the CI path)."""
    history = load_history(cfg.history_path)

    if not rebuild_only:
        todo = dates if dates is not None else pending_dates(load_calendar(cfg.calendar_path), history)
        if todo:
            kw = {} if get_html is None else {"get_html": get_html}
            statements = fetch_statements(todo, cfg.statements_dir, **kw)
            if not statements.empty:
                clf = roberta if roberta is not None else build_classifier(cfg, use_roberta)
                scored = score_all_measures(statements, lexicon_dir=cfg.lexicon_dir, roberta=clf)
                history = upsert_history(history, scored)
                save_history(history, cfg.history_path)
                _write_latest_redline(cfg, history)
            else:
                logger.warning("no statements fetched for %d pending date(s)", len(todo))

    render_site(history, tone_deltas(history), _load_segments(cfg.redline_path),
                cfg.site_out, verdict_url=VERDICT_URL)
    logger.info("dashboard written to %s (%d statements)", cfg.site_out, len(history))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m cbp.monitor",
                                 description="FOMC statement monitor → static dashboard")
    ap.add_argument("--date", help="process a single statement date YYYY-MM-DD (else all pending)")
    ap.add_argument("--no-roberta", action="store_true", help="torch-free run; roberta_stance=NaN")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="re-render HTML from the committed CSV + redline JSON (no fetch, no torch)")
    args = ap.parse_args()
    dates = [dt.date.fromisoformat(args.date)] if args.date else None
    run_monitor(Config(), dates=dates, use_roberta=not args.no_roberta,
                rebuild_only=args.rebuild_only)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_monitor_main.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/__main__.py tests/test_monitor_main.py
git commit -m "feat(monitor): orchestrator + CLI (--date/--no-roberta/--rebuild-only)"
```

---

### Task 8: Seed the committed FOMC calendar

**Files:**
- Create: `scripts/seed_calendar.py`
- Create: `data/monitor/fomc_calendar.csv` (generated, then committed)
- Test: `tests/test_monitor_calendar.py` (append a guard on the real file)

- [ ] **Step 1: Write the failing test (real-file guard)**

```python
# tests/test_monitor_calendar.py — APPEND
from pathlib import Path


def test_repo_calendar_exists_and_covers_recent_meetings():
    cal = load_calendar(Path("data/monitor/fomc_calendar.csv"))
    assert len(cal) >= 180                      # 1999→2026 ≈ 8/yr
    assert dt.date(2026, 6, 17) in cal          # last RECENT_DATES entry
    assert min(cal).year <= 1999 and max(cal).year >= 2026
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_calendar.py::test_repo_calendar_exists_and_covers_recent_meetings -v`
Expected: FAIL with `ValueError: FOMC calendar not found at data/monitor/fomc_calendar.csv`

- [ ] **Step 3: Write the seed script**

```python
# scripts/seed_calendar.py
"""One-time generator for data/monitor/fomc_calendar.csv.

Unions the Bauer-Swanson announcement dates (1999-2023, from the surprise xlsx) with
the post-2023 meeting dates the BS file doesn't cover, and writes a single sorted
`date` column. This file REPLACES the hand-maintained RECENT_DATES list in
scripts/action_tone_monitor.py — extend POST_2023 once a year from the Fed calendar.

Usage: python scripts/seed_calendar.py   (needs the BS xlsx under data/raw/; .[dev])
"""
from pathlib import Path

import pandas as pd

from cbp.config import Config
from cbp.data.mp_surprise import load_surprise

POST_2023 = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
    "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18",
    "2026-04-29", "2026-06-17",
]


def main() -> None:
    cfg = Config()
    su = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                       sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    su = su[su["date"].dt.year >= 1999]
    dates = sorted({d.date() for d in su["date"]} | {pd.Timestamp(s).date() for s in POST_2023})
    out = Path("data/monitor/fomc_calendar.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": [d.isoformat() for d in dates]}).to_csv(out, index=False)
    print(f"wrote {out} with {len(dates)} meeting dates ({dates[0]} … {dates[-1]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the calendar and verify the test passes**

Run: `python scripts/seed_calendar.py && pytest tests/test_monitor_calendar.py -v`
Expected: prints the row count (~197+20), then PASS (5 tests). The `data/monitor/fomc_calendar.csv` is created.

- [ ] **Step 5: Commit (force-add — `/data/*` is gitignored, the negation in Task 1 re-includes the dir)**

```bash
git add scripts/seed_calendar.py tests/test_monitor_calendar.py data/monitor/fomc_calendar.csv
git status --short   # confirm data/monitor/fomc_calendar.csv is staged (not ignored)
git commit -m "feat(monitor): seed committed FOMC calendar (replaces RECENT_DATES)"
```

---

### Task 9: CI workflow + retire `action_tone_monitor.py`

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `scripts/action_tone_monitor.py` (replace body with a deprecation pointer)

- [ ] **Step 1: Write the CI workflow**

```yaml
# .github/workflows/pages.yml
name: Publish FOMC monitor dashboard
on:
  push:
    branches: [main]
    paths:
      - "data/monitor/tone_history.csv"
      - "data/monitor/latest_redline.json"
      - "src/cbp/monitor/**"
      - ".github/workflows/pages.yml"
  workflow_dispatch: {}
permissions:
  contents: write   # peaceiris pushes to the gh-pages branch
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install (torch-free — site extra only)
        run: pip install -e ".[site]"
      - name: Render dashboard from committed artifacts
        run: python -m cbp.monitor --rebuild-only
      - name: Publish to gh-pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          publish_branch: gh-pages
```

- [ ] **Step 2: Retire the superseded script**

Replace the entire body of `scripts/action_tone_monitor.py` with:

```python
"""DEPRECATED — superseded by the monitor engine.

The descriptive hike/cut/hold action tracker is now the `action` column produced by
`python -m cbp.monitor`, and the hand-maintained meeting-date list lives in
`data/monitor/fomc_calendar.csv` (generated by scripts/seed_calendar.py). Run:

    python -m cbp.monitor                # score new statements + rebuild the dashboard
    python -m cbp.monitor --rebuild-only # re-render only (the CI path)

This shim is kept so old references fail loudly instead of running stale logic.
"""
import sys

if __name__ == "__main__":
    sys.exit("action_tone_monitor.py is deprecated — use `python -m cbp.monitor` (see module docstring).")
```

- [ ] **Step 3: Verify the suite is still green and the shim exits non-zero**

Run: `pytest -q && python scripts/action_tone_monitor.py; echo "exit=$?"`
Expected: full suite PASS; the shim prints the deprecation message and `exit=1`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pages.yml scripts/action_tone_monitor.py
git commit -m "ci(monitor): gh-pages publish workflow; retire action_tone_monitor.py"
```

---

### Task 10: Wire the real verdict URL, README usage, context docs, full green

**Files:**
- Modify: `src/cbp/monitor/site.py:11` (`VERDICT_URL`)
- Modify: `README.md` (Usage section)
- Modify: `docs/context/todo.md`, `docs/context/results.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: Set the real verdict URL**

```python
# src/cbp/monitor/site.py — replace the VERDICT_URL line
VERDICT_URL = "https://github.com/AlanVaa/Central_Bank_Policy/blob/main/docs/results/2026-06-29-lexicon-baseline-verdict.md"
```
(If the remote slug differs, use `git remote get-url origin` to confirm owner/repo first.)

- [ ] **Step 2: Add a README usage block**

```markdown
<!-- README.md — append under "## Usage" -->

### Statement monitor → dashboard (descriptive, not predictive)

```bash
python -m cbp.monitor                 # score new statements + rebuild dashboard (needs .[infer,site])
python -m cbp.monitor --no-roberta    # torch-free fast run (RoBERTa column gapped)
python -m cbp.monitor --rebuild-only  # re-render HTML from committed CSV (the CI path; .[site] only)
```

Each run upserts `data/monitor/tone_history.csv` + `data/monitor/latest_redline.json` (commit both);
CI (`.github/workflows/pages.yml`) re-renders and publishes to the `gh-pages` branch. Extend the
meeting list yearly in `data/monitor/fomc_calendar.csv` (`scripts/seed_calendar.py`).
```

- [ ] **Step 3: Update context docs (one line each)**

```markdown
<!-- docs/context/results.md — prepend a dated line -->
- [2026-06-30] Statement monitor SHIPPED (feat/statement-monitor, PRD 004). `python -m cbp.monitor`: fetch→score(action+lexicon+RoBERTa)→upsert tone_history.csv + latest_redline.json→render self-contained Plotly dashboard (levels, meeting-over-meeting deltas, latest-vs-prior redline). Torch-free CI (`--rebuild-only`) publishes to gh-pages. Descriptive-only (carries Phase 1/2a NO-GO banner). Calendar CSV replaces RECENT_DATES; action_tone_monitor.py retired.
```
```markdown
<!-- docs/context/todo.md — under open work, replace the stale action-tracker note with -->
Monitor (PRD 004) DONE on feat/statement-monitor. Open: [pending] merge → main; [pending] extend fomc_calendar.csv each Dec from the Fed calendar.
```
```markdown
<!-- docs/context/sesion-log.md — append -->
- [2026-06-30]: Built FOMC statement monitor + static gh-pages Plotly dashboard (descriptive). Brainstorm→PRD 004→plan→TDD build.
```

- [ ] **Step 4: Full suite + a live smoke render**

Run: `pytest -q`
Expected: PASS (all prior + new monitor tests).

Run (smoke, needs cached statements + `.[site]`): `python -m cbp.monitor --rebuild-only && test -f site/index.html && echo OK`
Expected: `OK` (renders from the committed CSV without fetching or torch).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/site.py README.md docs/context/results.md docs/context/todo.md docs/context/sesion-log.md
git commit -m "docs(monitor): verdict link, README usage, context log"
```

---

## Self-review checklist (completed)

- **Spec coverage:** engine command (T7) · tone_history contract (T2) · calendar replaces RECENT_DATES (T3,T8) · all-three scoring incl. graceful RoBERTa degrade (T4) · deltas+redline (T5) · 4-panel Plotly page + banner (T6) · gh-pages CI, torch-free rebuild (T9) · CLI flags (T7) · supersede action_tone_monitor.py (T9) · README/docs (T10). All §-sections mapped.
- **Placeholder scan:** none — every code/test step is complete; `VERDICT_URL` starts as a real placeholder string and is set to the actual link in T10.
- **Type consistency:** `HISTORY_COLUMNS` defined in T2, imported by T4; `score_all_measures(statements, *, lexicon_dir, roberta=)`, `tone_deltas(history)->dict`, `redline(prev,curr)->list[dict]`, `render_site(history, deltas, segments, out_path, *, verdict_url=)`, `run_monitor(cfg, *, dates, use_roberta, rebuild_only, get_html, roberta)` — signatures match across T4–T7. RoBERTa injection seam mirrors the existing `score_statements(statements, classifier)` contract.
- **Deviations** recorded in the header (committed `latest_redline.json`; `--backfill` dropped).
