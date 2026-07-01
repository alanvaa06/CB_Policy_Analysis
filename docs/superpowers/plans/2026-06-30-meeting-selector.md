# Meeting Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dropdown to the FOMC tracker that swaps the deltas table + redline to any meeting vs its immediate prior, and drops a dotted marker on each time-series chart at the selected date.

**Architecture:** On a full run, precompute for every consecutive statement pair a pre-rendered deltas-table HTML + redline HTML, written to a committed `site/redlines.json` (`{date: {deltas_html, redline_html}}`). The page renders the latest pair inline on load (unchanged), plus a `<select>` and small JS that fetches `redlines.json` once on first non-default selection, swaps two `innerHTML`s, and calls `Plotly.relayout` to move a vertical marker. CI/`--rebuild-only` renders from the committed CSV + `redlines.json` with no raw texts and no torch.

**Tech Stack:** Python 3, pandas, plotly, difflib; pytest; vanilla JS in the generated static page.

**Design refinement vs spec:** deltas are shipped as pre-rendered HTML *inside* `redlines.json` (not inlined in the page) to keep `index.html` small and avoid inline git churn. Same approved architecture: separate JSON fetched on demand, pre-rendered HTML as single source of truth.

> **Corrections applied during execution (see final commits):**
> 1. **Deploy path.** `/site/` is gitignored and CI runs `--rebuild-only` (no raw texts), so the payload could not live at a committed `site/redlines.json`. Shipped instead at **`data/monitor/redlines.json`** (committed, like `latest_redline.json`), which `run_monitor` **copies into `./site` after render** (both full and rebuild-only). `Config.redlines_path = data/monitor/redlines.json`; added to `pages.yml` trigger paths. Everywhere Task 4 / Task 5 below say `site/redlines.json`, the committed source is `data/monitor/redlines.json`.
> 2. **Regeneration gating.** `_write_all_redlines` was moved OUT of the "new statements fetched" branch to run on **every non-rebuild run**, so the artifact bootstraps/refreshes even when there are no pending meetings.
> 3. **Fetch hardening.** `_loadPayload` wraps `fetch` in try/catch and `_onSelect` guards `map && map[date]`, so a missing/404 payload degrades gracefully (marker still moves).

---

## File structure

- **Modify** `src/cbp/monitor/contrast.py` — extract `_pair_deltas`, add `all_pair_deltas(history)`.
- **Modify** `src/cbp/monitor/site.py` — explicit `div_id`s on figures; `build_redlines_payload`; dropdown + slot ids + swap/marker JS in `render_site`.
- **Modify** `src/cbp/monitor/__main__.py` — `_write_all_redlines`; wire into `run_monitor`.
- **Modify** `src/cbp/config.py` — add `redlines_path = Path("site/redlines.json")`.
- **Modify** tests: `tests/test_monitor_metrics.py`? no — new/edited tests in `tests/test_monitor_site.py`, `tests/test_monitor_main.py`, `tests/test_monitor_metrics.py` (deltas live in contrast; use `tests/test_monitor_score.py`? deltas are tested where `tone_deltas` is). Add contrast tests to `tests/test_monitor_site.py` is wrong — create `tests/test_contrast_deltas.py`.
- **Generated (committed)** `site/redlines.json`, regenerated `site/index.html`.

---

### Task 1: `all_pair_deltas` in contrast.py

**Files:**
- Modify: `src/cbp/monitor/contrast.py`
- Test: `tests/test_contrast_deltas.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrast_deltas.py
import pandas as pd
from cbp.monitor.contrast import tone_deltas, all_pair_deltas


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01"]),
        "action": [1.0, -1.0, 0.0],
        "lexicon_tone": [0.5, 0.2, 0.0],
        "roberta_stance": [0.4, 0.1, float("nan")],
    })


def test_all_pair_deltas_keyed_by_latest_date():
    m = all_pair_deltas(_hist())
    assert set(m) == {"2024-03-20", "2024-05-01"}          # every date except the first


def test_all_pair_deltas_matches_tone_deltas_on_last_pair():
    hist = _hist()
    assert all_pair_deltas(hist)["2024-05-01"] == tone_deltas(hist)


def test_all_pair_deltas_none_delta_on_nan():
    m = all_pair_deltas(_hist())
    assert m["2024-05-01"]["roberta_stance"]["delta"] is None  # nan latest -> None


def test_all_pair_deltas_empty_when_single_row():
    one = _hist().iloc[:1]
    assert all_pair_deltas(one) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contrast_deltas.py -v`
Expected: FAIL — `ImportError: cannot import name 'all_pair_deltas'`.

- [ ] **Step 3: Refactor `tone_deltas` and add `all_pair_deltas`**

In `src/cbp/monitor/contrast.py`, replace the body of `tone_deltas` and add the two helpers. Keep `_num`, `MEASURES` as-is.

```python
def _pair_deltas(prior: pd.Series, latest: pd.Series) -> dict:
    """Change per measure between two history rows (prior -> latest)."""
    out = {
        "date_prior": pd.Timestamp(prior["date"]).strftime("%Y-%m-%d"),
        "date_latest": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
    }
    for m in MEASURES:
        p, l = _num(prior[m]), _num(latest[m])
        out[m] = {"prior": p, "latest": l,
                  "delta": (None if p is None or l is None else l - p)}
    return out


def tone_deltas(history: pd.DataFrame) -> dict:
    """Latest-vs-prior change per measure, from the last two history rows.
    Returns {} when fewer than two rows."""
    if len(history) < 2:
        return {}
    return _pair_deltas(history.iloc[-2], history.iloc[-1])


def all_pair_deltas(history: pd.DataFrame) -> dict:
    """`_pair_deltas` for every consecutive pair, keyed by the latest date string.
    {} when fewer than two rows."""
    if len(history) < 2:
        return {}
    out = {}
    for i in range(1, len(history)):
        d = _pair_deltas(history.iloc[i - 1], history.iloc[i])
        out[d["date_latest"]] = d
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contrast_deltas.py tests/test_monitor_site.py -v`
Expected: PASS (existing `tone_deltas` behavior preserved).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/contrast.py tests/test_contrast_deltas.py
git commit -m "feat(tracker): all_pair_deltas — per-pair deltas keyed by latest date"
```

---

### Task 2: `build_redlines_payload` in site.py

**Files:**
- Modify: `src/cbp/monitor/site.py`
- Test: `tests/test_monitor_site.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_monitor_site.py`)

```python
def test_build_redlines_payload_shape():
    from cbp.monitor.site import build_redlines_payload
    deltas_by_date = {
        "2024-05-01": {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
                       "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
                       "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
                       "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}},
    }
    segs_by_date = {"2024-05-01": [{"op": "replace", "prev": "hold", "curr": "raise"}]}
    payload = build_redlines_payload(deltas_by_date, segs_by_date)
    assert set(payload) == {"2024-05-01"}
    entry = payload["2024-05-01"]
    assert "deltas" in entry["deltas_html"].lower()      # <table class='deltas'>
    assert "rl-insert" in entry["redline_html"]          # rendered redline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_site.py::test_build_redlines_payload_shape -v`
Expected: FAIL — `ImportError: cannot import name 'build_redlines_payload'`.

- [ ] **Step 3: Add `build_redlines_payload`** to `src/cbp/monitor/site.py` (after `_deltas_table_html`)

```python
def build_redlines_payload(deltas_by_date: dict, segments_by_date: dict) -> dict:
    """For every date present in `segments_by_date`, pre-render both panels.
    Returns {date: {"deltas_html": ..., "redline_html": ...}} — the toggle payload
    served as site/redlines.json. Both renderers stay the single source of truth."""
    payload = {}
    for date, segments in segments_by_date.items():
        payload[date] = {
            "deltas_html": _deltas_table_html(deltas_by_date.get(date, {})),
            "redline_html": build_redline_html(segments),
        }
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitor_site.py::test_build_redlines_payload_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/site.py tests/test_monitor_site.py
git commit -m "feat(tracker): build_redlines_payload — pre-rendered deltas+redline per date"
```

---

### Task 3: Dropdown, slot ids, chart div_ids, and swap JS in `render_site`

**Files:**
- Modify: `src/cbp/monitor/site.py` (`render_site`, `_PAGE`, `_CSS`, figure `to_html` calls)
- Test: `tests/test_monitor_site.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_monitor_site.py`)

```python
def test_render_site_has_meeting_selector_and_hooks(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist_v2(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="meeting"' in html                          # the dropdown
    assert html.count("<option") == 2                      # 3 rows -> 2 pairs w/ a prior
    assert 'value="2024-05-01"' in html                    # newest option present
    assert 'id="deltas-slot"' in html and 'id="redline-slot"' in html
    for cid in ("fig-heatmap", "fig-change", "fig-commstyle", "fig-levels", "fig-deltas"):
        assert cid in html                                 # explicit chart div ids
    assert "redlines.json" in html                         # fetch target
    assert "Plotly.relayout" in html                       # marker hook
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_site.py::test_render_site_has_meeting_selector_and_hooks -v`
Expected: FAIL — no `id="meeting"` in output.

- [ ] **Step 3: Edit `render_site`, `_PAGE`, `_CSS`, and figure `to_html` calls**

In `src/cbp/monitor/site.py`:

(a) Give each figure an explicit `div_id` in `render_site`:

```python
def render_site(history: pd.DataFrame, deltas: dict, segments: list[dict],
                out_path: Path, *, verdict_url: str = VERDICT_URL,
                redlines_url: str = "redlines.json") -> None:
    """Assemble the tracker and write it to `out_path`. When history has >=2 rows,
    a meeting selector is emitted; picking a date swaps the deltas table + redline
    (from `redlines_url`, fetched once) and moves a marker on each chart."""
    heatmap = build_theme_heatmap(history).to_html(full_html=False, include_plotlyjs="inline", div_id="fig-heatmap")
    change_fig = build_change_magnitude_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-change")
    commstyle = build_commstyle_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-commstyle")
    levels = build_levels_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-levels")
    deltas_fig = build_deltas_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-deltas")
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in history["date"]]
    selector = _selector_html(dates)
    page = _PAGE.format(css=_CSS, glossary=glossary_html(), selector=selector,
                        deltas_table=_deltas_table_html(deltas),
                        redline=build_redline_html(segments), heatmap=heatmap, change_fig=change_fig,
                        commstyle=commstyle, levels=levels, deltas_fig=deltas_fig,
                        toggle_js=_toggle_js(redlines_url))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
```

(b) Add the selector + JS builders (after `_deltas_table_html`):

```python
def _selector_html(dates: list[str]) -> str:
    """Dropdown of every meeting that has a prior (all dates except the first),
    newest first, latest pre-selected. Empty string when <2 meetings."""
    pairs = dates[1:]
    if not pairs:
        return ""
    opts = "".join(f'<option value="{d}"{" selected" if d == pairs[-1] else ""}>{d}</option>'
                   for d in reversed(pairs))
    return ('<p class="selector">Show statement: '
            f'<select id="meeting">{opts}</select> <span>vs its prior</span></p>')


def _toggle_js(redlines_url: str) -> str:
    """Client-side swap: fetch the payload once, replace deltas+redline innerHTML,
    and move a dotted vertical marker on each chart via Plotly.relayout."""
    ids = ["fig-heatmap", "fig-change", "fig-commstyle", "fig-levels", "fig-deltas"]
    return """
<script>
const CHART_IDS = %s;
let _payload = null;
async function _loadPayload(){
  if(!_payload){ _payload = await fetch("%s").then(r => r.json()); }
  return _payload;
}
function _mark(date){
  const shapes = [{type:"line", xref:"x", yref:"paper", x0:date, x1:date,
                   y0:0, y1:1, line:{color:"#888", dash:"dot", width:1}}];
  CHART_IDS.forEach(id => {
    const el = document.getElementById(id);
    if(el && window.Plotly){ Plotly.relayout(el, {shapes: shapes}); }
  });
}
async function _onSelect(date){
  const map = await _loadPayload();
  const entry = map[date];
  if(entry){
    document.getElementById("deltas-slot").innerHTML = entry.deltas_html;
    document.getElementById("redline-slot").innerHTML = entry.redline_html;
  }
  _mark(date);
}
document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("meeting");
  if(sel){
    sel.addEventListener("change", e => _onSelect(e.target.value));
    _mark(sel.value);   // marker for the default (latest); no fetch needed
  }
});
</script>""" % (json.dumps(ids), redlines_url)
```

(c) Add `import json` at the top of `site.py` (next to `import html as _html`).

(d) Update `_PAGE` — add `{selector}`, wrap the two panels in slots, add `{toggle_js}` before `</body>`:

```python
_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FOMC Statement Tracker</title><style>{css}</style></head><body>
<h1>FOMC Statement Tracker</h1>
<h2>How to read this tracker</h2>
<div class="banner">{glossary}</div>
<h2>Latest statement — what changed vs the prior</h2>
{selector}
<div id="deltas-slot">{deltas_table}</div>
<div id="redline-slot" class="redline">{redline}</div>
<h2>What the Fed is focused on</h2>{heatmap}
<h2>How much each statement changed</h2>{change_fig}
<h2>Communication style</h2>{commstyle}
<h2>Stance measures, in context</h2>{levels}{deltas_fig}
{toggle_js}
</body></html>"""
```

(e) Add selector styling to `_CSS` (append inside the triple-quoted string):

```python
.selector{font-size:14px;margin:6px 0}.selector select{font-size:14px;padding:2px 4px}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitor_site.py -v`
Expected: PASS — new test passes; the existing `test_render_site_*` tests still pass (they don't assert on the selector, and slots keep the same text content).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/site.py tests/test_monitor_site.py
git commit -m "feat(tracker): meeting selector + chart marker in render_site"
```

---

### Task 4: `redlines_path` config + `_write_all_redlines` wired into `run_monitor`

**Files:**
- Modify: `src/cbp/config.py`
- Modify: `src/cbp/monitor/__main__.py`
- Test: `tests/test_monitor_main.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_monitor_main.py`)

```python
def test_run_monitor_writes_all_redlines_json(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.calendar_path.write_text("date\n2024-01-31\n2024-03-20\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)
    payload = json.loads(cfg.redlines_path.read_text(encoding="utf-8"))
    assert set(payload) == {"2024-03-20"}                 # one pair -> keyed by latest date
    entry = payload["2024-03-20"]
    assert entry["redline_html"] and entry["deltas_html"]
    # page references the JSON and offers the selector
    html = cfg.site_out.read_text(encoding="utf-8")
    assert 'id="meeting"' in html and "redlines.json" in html


def test_rebuild_only_reuses_committed_redlines(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.calendar_path.write_text("date\n2024-01-31\n2024-03-20\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)
    cfg.site_out.unlink()
    def _boom(url):
        raise AssertionError("rebuild-only must not fetch")
    m.run_monitor(cfg, rebuild_only=True, get_html=_boom)   # no regeneration of redlines.json
    assert cfg.site_out.exists()
    assert 'id="meeting"' in cfg.site_out.read_text(encoding="utf-8")
```

Update `_cfg` in this file to set `redlines_path`:

```python
def _cfg(tmp_path) -> Config:
    return Config(
        history_path=tmp_path / "tone_history.csv",
        calendar_path=tmp_path / "cal.csv",
        redline_path=tmp_path / "latest_redline.json",
        redlines_path=tmp_path / "site" / "redlines.json",
        statements_dir=tmp_path / "statements",
        site_out=tmp_path / "site" / "index.html",
        lexicon_dir=__import__("pathlib").Path("data/lexicons"),
        themes_path=__import__("pathlib").Path("data/lexicons/themes.json"),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_main.py::test_run_monitor_writes_all_redlines_json -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'redlines_path'`.

- [ ] **Step 3a: Add `redlines_path` to `Config`**

In `src/cbp/config.py`, after the `redline_path` line:

```python
    redlines_path: Path = Path("site/redlines.json")
```

- [ ] **Step 3b: Add `_write_all_redlines` and wire it into `run_monitor`**

In `src/cbp/monitor/__main__.py`:

Add imports at top (extend existing):

```python
from cbp.monitor.contrast import all_pair_deltas, redline, tone_deltas
from cbp.monitor.site import VERDICT_URL, build_redlines_payload, render_site
```

Add the writer (after `_write_latest_redline`):

```python
def _write_all_redlines(cfg: Config, history: pd.DataFrame) -> None:
    """Precompute the toggle payload for every consecutive pair and write it to
    cfg.redlines_path ({date: {deltas_html, redline_html}}). Reads all statement
    texts from the local cache; full-run only (CI lacks the raw texts)."""
    if len(history) < 2:
        return
    dates = [pd.Timestamp(d).date() for d in history["date"]]
    texts = fetch_statements(dates, cfg.statements_dir)   # cache hit; no network locally
    tmap = {pd.Timestamp(r.date).date(): r.text for r in texts.itertuples()}
    segments_by_date = {}
    for i in range(1, len(dates)):
        prev_d, curr_d = dates[i - 1], dates[i]
        if prev_d in tmap and curr_d in tmap:
            key = curr_d.strftime("%Y-%m-%d")
            segments_by_date[key] = redline(tmap[prev_d], tmap[curr_d])
    payload = build_redlines_payload(all_pair_deltas(history), segments_by_date)
    Path(cfg.redlines_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.redlines_path).write_text(json.dumps(payload), encoding="utf-8")
```

In `run_monitor`, inside the `if not rebuild_only:` block, right after `_write_latest_redline(cfg, history)`:

```python
                _write_latest_redline(cfg, history)
                _write_all_redlines(cfg, history)
```

The final `render_site(...)` call is unchanged — `render_site` now derives the dropdown from `history` itself and references `redlines.json` (relative to `site/index.html`) by default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitor_main.py -v`
Expected: PASS — both new tests pass; existing end-to-end and rebuild-only tests still pass (`_write_latest_redline` and default render unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/config.py src/cbp/monitor/__main__.py tests/test_monitor_main.py
git commit -m "feat(tracker): write site/redlines.json for all pairs; wire selector into run_monitor"
```

---

### Task 5: Regenerate the live site and commit generated assets

**Files:**
- Generated: `site/index.html`, `site/redlines.json`

- [ ] **Step 1: Full test suite green**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 2: Regenerate the dashboard from committed data (torch-free)**

Run: `python -m cbp.monitor --rebuild-only` first to confirm the CI path renders, then a full local rebuild to (re)generate `redlines.json` from the raw statement cache:

Run: `python -m cbp.monitor --no-roberta`
Expected: logs "dashboard written to site/index.html (227 statements)"; `site/redlines.json` created.

- [ ] **Step 3: Manual smoke check**

Open `site/index.html` in a browser. Verify:
- dropdown lists meetings newest-first, latest selected;
- selecting an earlier meeting (e.g. a June 2025 date) swaps the deltas table + redline;
- a dotted vertical marker appears on the charts at the selected date (best-effort per known risk).

- [ ] **Step 4: Commit generated assets**

```bash
git add site/index.html site/redlines.json
git commit -m "chore(tracker): regenerate dashboard with meeting selector + redlines.json"
```

---

## Self-review notes

- **Spec coverage:** deltas map → Task 1; pre-rendered payload → Task 2; dropdown + slots + marker + chart div_ids → Task 3; `redlines_path` + `_write_all_redlines` + CI path → Task 4; regenerate/commit assets → Task 5. Marker known-risk documented in Task 5 step 3.
- **Type consistency:** payload shape `{date: {deltas_html, redline_html}}` used identically in Task 2 (`build_redlines_payload`), Task 4 (writer + assertions), and Task 3 JS (`entry.deltas_html`/`entry.redline_html`). `all_pair_deltas` keys = latest-date strings, matching `segments_by_date` keys and dropdown option values. Chart id list identical in Task 3 `render_site`, `_toggle_js`, and the Task 3 test.
- **Backward compat:** `render_site` signature adds only keyword-defaulted params; existing 4-positional-arg call sites and tests unaffected. `_write_latest_redline` + `latest_redline.json` untouched, so the default on-load render and its tests are unchanged.
- **No placeholders:** every code step is complete.
```