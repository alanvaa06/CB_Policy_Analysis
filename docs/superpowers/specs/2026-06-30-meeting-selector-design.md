# Design — meeting selector for the FOMC statement tracker

**Date:** 2026-06-30
**Status:** implemented (branch `feat/meeting-selector`)

> **Correction (2026-07-01, discovered during implementation).** This spec below
> says the payload is committed to `site/redlines.json`. That is WRONG for this repo:
> `/site/` is gitignored and CI (`.github/workflows/pages.yml`) publishes via
> `python -m cbp.monitor --rebuild-only` (torch-free, **no raw statement texts**), so
> `site/redlines.json` can neither be committed nor regenerated in CI. **What shipped:**
> the payload lives at `data/monitor/redlines.json` (committed, following the
> `latest_redline.json` pattern) and `run_monitor` **copies it into `./site` after
> render** (both full and `--rebuild-only` runs) so gh-pages serves it. `Config.redlines_path`
> defaults to `data/monitor/redlines.json`; `data/monitor/redlines.json` is added to the
> pages.yml trigger paths. Read "`site/redlines.json` (committed)" below as
> "`data/monitor/redlines.json` (committed), published into `./site` at build."

## Problem

The dashboard (`site/index.html`) is hardcoded to compare the **latest** statement
vs its immediate prior. Both the deltas table (`tone_deltas`,
[contrast.py](../../../src/cbp/monitor/contrast.py)) and the redline
([__main__.py `_write_latest_redline`](../../../src/cbp/monitor/__main__.py)) only
ever use the last two rows of history. A user who wants to inspect an earlier
meeting (e.g. "June 2025 vs the previous one") cannot.

## Goal

Add a **meeting selector** (dropdown) to the tracker. Picking a meeting shows that
statement vs its **immediate prior** across three surfaces:

1. the "what changed vs prior" **deltas table**,
2. the word-level **redline**,
3. a dotted **vertical marker** at the selected date on each time-series chart.

Default on page load = latest meeting (current behavior preserved). Time-series
charts stay full-history; only a marker moves.

## Non-goals

- Arbitrary non-consecutive pairs (A vs B). Consecutive-only (meeting vs immediate prior).
- Truncating/zooming charts to the selected meeting. Charts stay full-history + marker.
- New statement fetching or scoring. Pure presentation over existing data.

## Architecture

### 1. Data build (Python, full-run only — requires raw statement texts)

Raw statement HTMLs live in `data/raw/statements/` locally but are **not**
git-tracked, so redline text can only be generated on a full run (not in CI).
Therefore all derived data is generated on the full run and **committed** so
`--rebuild-only`/CI can render without raw texts.

For every consecutive pair in `tone_history` (225 pairs):

- **Deltas map** — generalize `tone_deltas` to accept any (prior_row, latest_row)
  pair; build `{ "<latest_date>": { "date_prior", "date_latest",
  "<measure>": {prior, latest, delta} } }` for all pairs. Small. Inlined into the
  page as a JS object.
- **Redline** — run existing `redline(prev_text, curr_text)` per pair, render via
  existing `build_redline_html(segments)`, store the **pre-rendered HTML string**
  keyed by latest-date. Written to `site/redlines.json` (committed).

**Decision — pre-rendered HTML over raw segments:** storing rendered HTML keeps the
punctuation-aware join (`_sep` in site.py, `_smart_join` in contrast.py) as a single
source of truth in Python. The frontend only does `innerHTML =`; no porting the
spacing rules to JavaScript.

### 2. Delivery

- `site/redlines.json` (~2–6MB) — a `{ "<latest_date>": "<html>" }` map. Fetched
  **once on first non-default selection**, then cached in a JS variable. Committed
  so CI/`--rebuild-only` works without raw texts. On page load the latest redline is
  already rendered inline (current behavior), so no fetch is needed for the default view.
- **Deltas map** — inlined into `index.html` as a JS object (small enough).
- **Dropdown** — lists all meeting dates that have a prior (i.e. every date except
  the first), newest first, default-selected = latest.

### 3. Frontend (added to `site.py`)

- `<select id="meeting">` rendered above the deltas table.
- On `change`:
  1. rewrite the deltas-table container `innerHTML` from the inline deltas map,
  2. set the redline container `innerHTML` from the fetched/cached `redlines.json`
     (fetch-once, then lookup),
  3. `Plotly.relayout(divId, { shapes: [dotted vertical line at selected date] })`
     on each time-series chart.
- Requires giving each Plotly figure an **explicit `div_id`** (passed to `to_html`)
  so the JS can target them deterministically. IDs collected into a JS array.

### 4. CI / rebuild-only

- Rename/extend `_write_latest_redline` → `_write_all_redlines`: writes the full
  `{date: html}` map to `site/redlines.json` (and keeps the latest inline for
  default render). `run_monitor` passes the full map into `render_site`.
- `--rebuild-only` reads committed `site/redlines.json` + `tone_history.csv`; no raw
  texts, no torch. Same contract as today, wider payload.
- `Config`: add `redlines_path = Path("site/redlines.json")` (supersedes
  `redline_path`/`latest_redline.json`; keep old field until callers migrate).

## Data flow

```
full run:  fetch/score → upsert history CSV
            → for each consecutive pair: tone_deltas + redline + build_redline_html
            → deltas map (inline)  +  site/redlines.json (committed)
            → render_site(history, deltas_map, redlines_map) → site/index.html

rebuild-only / CI:  history CSV + committed site/redlines.json
            → render_site(...) → site/index.html   (no raw texts, no torch)
```

## Testing (TDD, per project standards)

- **all-pairs deltas map** — each entry matches per-pair `tone_deltas` on the same
  two rows; None/NaN (e.g. `--no-roberta` gaps) render as `—` and `delta=None`.
- **redlines.json** — keyed by every non-first date; each value is non-empty HTML;
  latest key equals today's single-pair redline.
- **page** — contains `<select id="meeting">` with N options (= pairs), the swap JS,
  the inline deltas map, and explicit chart `div_id`s.
- **rebuild-only** — renders a valid page from committed CSV + `redlines.json` alone
  (no `data/raw/statements`).

## Known risk (flagged)

The chart marker on the 3-row comm-style subplot and the heatmap needs correct axis
references (`xref`/`yref`). The dotted vertical marker is **best-effort** across all
charts; if subplot axis wiring proves fiddly, the marker degrades to the single-axis
charts and is documented. The **table + redline swap is the core deliverable and
always works** regardless of marker outcome.

## Decisions made during brainstorming

- Selector scope: **table + redline + chart marker** (not deltas-only, not full-page filter).
- Delivery: **separate `redlines.json`, fetched on demand** (not inline-everything, not per-meeting files).
- Redline payload: **pre-rendered HTML** (not raw segments) — single source of truth.
- Default load: **latest meeting** — preserves current view.
