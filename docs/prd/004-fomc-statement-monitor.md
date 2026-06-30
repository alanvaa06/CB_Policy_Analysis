# 004 — FOMC Statement Monitor → Static GitHub Pages Dashboard

**Status:** approved (design) · **Date:** 2026-06-30 · **Owner:** Alan
**Track:** descriptive analytics / monitoring (NOT a predictive signal) · **Builds on:** the cached statement-fetch + the three scorers (RoBERTa, hawk/dove lexicon, action lexicon) from Phases 1–2a; all three predictive verdicts = NO-GO.

## 1. Problem & goal

The repo can fetch every FOMC statement and score its tone three ways, but each new
statement is processed by a hand-edited one-off (`scripts/action_tone_monitor.py` with a
hand-maintained `RECENT_DATES` list) and the output is a single static PNG. There is no
repeatable "apply the analysis when a new statement drops" path, and no place to *see*
tone through time or compare a statement against its predecessor.

**Goal:** an idempotent **monitor engine** — one command — that, given the newest FOMC
statement, scores it on all three measures, appends to a committed tone history, and
regenerates a **static, interactive (Plotly) dashboard published to GitHub Pages**. The
headline view is the **latest-vs-prior contrast**: numeric tone deltas plus a
sentence-level **track-changes redline** of the wording. Tone **levels** and
**meeting-over-meeting deltas** are shown across full history.

**Framing (non-negotiable):** this is a **descriptive monitor**, not a trading signal.
Phases 1–2a established statement tone adds *no* out-of-sample value beyond the policy
surprise. The dashboard carries that verdict in a banner. Value here is analytics +
the redline, which markets read regardless of marginal predictive content.

Non-goal: prediction, alpha, auto-scheduling, a live server.

## 2. Scope

**In scope (v1):**
- A `cbp.monitor` engine command: discover pending statement dates → fetch → score all
  three measures → upsert tone history → render dashboard.
- A committed **tone history** (`data/monitor/tone_history.csv`) as the single source of
  truth, and a committed **meeting calendar** (`data/monitor/fomc_calendar.csv`) that
  **replaces** the hand-maintained `RECENT_DATES` list.
- A `contrast` module: `tone_deltas` (latest vs prior) + `redline` (sentence diff).
- A `site` module: Plotly **levels** + **deltas** figures and a **redline** panel rendered
  to one self-contained `index.html`.
- A **GitHub Action** that rebuilds the HTML from the committed CSV (torch-free) and
  publishes it to the `gh-pages` branch.
- TDD, offline (tests never import torch or hit the network).

**Out of scope (YAGNI):** auto-scheduler / cron, Streamlit or any server, new tone
measures or lexicons, predictive/signal output, building the HTML inside CI with the
heavy model, negation-aware diffing, statement *minutes*/pressers (statements only).

## 3. Architecture

New `src/cbp/monitor/` subpackage. Pure logic in modules (offline-testable); the only IO
shells are HTML rendering and file/network. Reuses the existing fetch + scorers untouched.

```
NEW:
  monitor/history.py    # load / idempotent-upsert / save tone_history.csv  (pure pandas)
  monitor/calendar.py   # load_calendar + pending_dates(calendar, history)  (pure)
  monitor/score.py      # score_all_measures(): action + lexicon + RoBERTa -> merged frame
  monitor/contrast.py   # tone_deltas(history) + redline(prev_text, curr_text)  (pure)
  monitor/site.py       # build_levels_figure / build_deltas_figure / render_site -> index.html
  monitor/__main__.py   # the engine command (orchestration + CLI flags)
  data/monitor/fomc_calendar.csv   # known meeting dates (data, extended yearly)
  data/monitor/tone_history.csv    # committed source of truth (built by the engine)
  .github/workflows/pages.yml      # torch-free rebuild + publish to gh-pages
EXTENDED:
  config.py             # monitor paths (history, calendar, site out); lexicon dir
  pyproject [site] extra  # + plotly
REUSED (untouched):
  data/fomc_statements.fetch_statements        # [date, text], cached HTML, skip-on-fail
  models/lexicon_scorer.{load_lexicon, score_statements_lexicon}  # action + hawk/dove
  models/stance_scorer.{load_fomc_roberta, score_statements, split_sentences}  # RoBERTa + sentence split
```

**The committed `tone_history.csv` is the contract between two runtimes:**
- **Local (heavy, manual):** the engine fetches + scores (incl. RoBERTa) and writes the
  CSV. User commits the CSV (and newly cached statement HTML).
- **CI (light, torch-free):** `--rebuild-only` reads the committed CSV and re-renders the
  HTML with Plotly, then publishes. CI never installs torch and never hits the network.

This keeps `index.html` a **derived artifact** (built in CI, not committed) and makes the
dashboard reproducible from the CSV alone.

## 4. Data contracts

- **Calendar** — `fomc_calendar.csv`: column `date` (YYYY-MM-DD), one row per scheduled
  FOMC announcement. Seeded by backfilling the Bauer-Swanson dates (1999–2023) +
  `RECENT_DATES` (2024–2026); extended by hand once a year (Fed publishes ~1yr ahead).
- **Tone history** — `tone_history.csv`, one row per statement, sorted by date:
  `date: YYYY-MM-DD, action: float, lexicon_tone: float, roberta_stance: float|NaN, n_sentences: int`.
  `action` ∈ {−1,0,+1} (cut/hold/hike). `lexicon_tone`, `roberta_stance` ∈ [−1,+1].
  `roberta_stance` is `NaN` for rows scored under `--no-roberta`.
- **Deltas** — `tone_deltas(history) -> {measure: {prior, latest, delta}}` over the last
  two rows, per measure (`delta = latest − prior`; `None` if either side missing/NaN).
- **Redline** — `redline(prev, curr) -> list[{op, prev, curr}]`, `op ∈
  {equal, insert, delete, replace}`, sentence-level (see §6).

Signatures (typed):
- `load_history(path: Path) -> pd.DataFrame` (empty frame with the schema if file absent)
- `upsert_history(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame` (by `date`,
  `new` overwrites a same-date row; result sorted, deduped)
- `save_history(history: pd.DataFrame, path: Path) -> None`
- `load_calendar(path: Path) -> list[dt.date]`
- `pending_dates(calendar: list[dt.date], history: pd.DataFrame) -> list[dt.date]`
- `score_all_measures(statements: pd.DataFrame, *, lexicon_dir: Path, roberta=None) -> pd.DataFrame`
- `tone_deltas(history: pd.DataFrame) -> dict`
- `redline(prev_text: str, curr_text: str) -> list[dict]`

## 5. Scoring orchestration (`score.py`)

`score_all_measures(statements, *, lexicon_dir, roberta=None)`:
1. `action` = `score_statements_lexicon(statements, *load_lexicon(lexicon_dir/"action_tone.json"))` → rename `stance`→`action`.
2. `lexicon_tone` = `score_statements_lexicon(statements, *load_lexicon(lexicon_dir/"hawk_dove.json"))` → rename `stance`→`lexicon_tone`.
3. `roberta_stance` = `score_statements(statements, roberta)` if `roberta` is not None, else `NaN` column + a logged warning. `n_sentences` from `split_sentences` per statement.
4. Merge on `date` → `[date, action, lexicon_tone, roberta_stance, n_sentences]`.

RoBERTa is **injected** (the loaded classifier is passed in), so tests pass a fake and
never import transformers — mirrors the existing `score_statements` test pattern. The
engine builds the real classifier lazily via `load_fomc_roberta` only when not
`--no-roberta`.

## 6. Contrast (`contrast.py`)

- `tone_deltas` — pull the last two history rows; per measure emit `{prior, latest, delta}`.
- `redline` — split both statements into sentences with the existing `split_sentences`
  (consistency with the RoBERTa path), then `difflib.SequenceMatcher` over the sentence
  lists. Map opcodes → segments: `equal` (unchanged), `insert` (new this statement),
  `delete` (dropped vs prior), `replace` (reworded). Pure stdlib, deterministic,
  fully unit-tested on crafted statement pairs. This is the "track-changes" view markets
  read; it is **textual, not semantic** (see §11).

## 7. Dashboard (`site.py` → `index.html`)

One self-contained HTML (Plotly `include_plotlyjs="inline"` → no external assets, opens
offline). Panels, top to bottom:
1. **Banner** — descriptive-monitor framing + link to the Phase 1/2a verdicts.
2. **Latest-vs-prior contrast** (headline): the meeting date, the three tone deltas
   (arrow + value), and the redline panel (added=green, dropped=struck/red,
   reworded=amber), rendered from `redline` segments.
3. **Tone levels through time** — action / lexicon / RoBERTa as lines, 1999→latest, with
   light era shading; honest gaps where a measure is silent (lexicon post-2024, RoBERTa
   under `--no-roberta`).
4. **Tone deltas through time** — meeting-over-meeting Δ per measure (bar/line), the
   delta-history.

Pure figure builders (`build_levels_figure`, `build_deltas_figure`,
`build_redline_html`) take the history / segments and return Plotly figures / HTML
fragments — unit-tested for trace structure on a tiny synthetic history. `render_site`
is the thin IO shell that assembles fragments + writes the file.

## 8. Publish (`.github/workflows/pages.yml`)

On push to `main` (paths: `data/monitor/tone_history.csv`, `src/cbp/monitor/**`) **and**
`workflow_dispatch`: checkout → `pip install -e ".[site]"` (plotly, **no torch**) →
`python -m cbp.monitor --rebuild-only` → publish the built site dir to `gh-pages`
(`peaceiris/actions-gh-pages` or `actions/deploy-pages`). Pages source = `gh-pages`
branch, so `docs/` (PRDs, verdicts) stays off the public site. CI is torch-free and
network-free beyond the checkout + pip.

## 9. CLI (`python -m cbp.monitor`)

- (default) — discover `pending_dates`, fetch, score all measures, upsert history, save
  CSV, render HTML; print the latest meeting + its deltas.
- `--date YYYY-MM-DD` — force-process one statement (also backfill a missed date).
- `--no-roberta` — torch-free fast run; `roberta_stance` column = `NaN` for new rows.
- `--rebuild-only` — re-render HTML from the existing committed CSV; no fetch, no scoring,
  no torch (the CI entry point).
- `--backfill` — process *all* calendar dates missing from history (one-time history build).

## 10. Error handling / edge cases

- A pending date whose statement 404s or parses empty → logged + skipped (reuse
  `fetch_statements` behavior); not written to history (no fabricated rows).
- Re-run with no new statements → engine is a no-op on the CSV, still re-renders HTML.
- Fewer than two history rows → contrast panel renders a "need ≥2 statements" placeholder
  (no crash).
- `roberta_stance` NaN rows → levels/deltas charts break the RoBERTa line at the gap
  rather than interpolating across it.
- Missing calendar or history file → `load_*` returns/raises clearly (empty history is
  valid on first run; missing calendar fails fast naming the path).
- Same-date re-score (e.g. a corrected statement) → `upsert_history` overwrites, never
  duplicates.

## 11. Stack & conventions

Typed Python + pandas; tests offline under `pythonpath=["src"]`, never importing torch or
the network (RoBERTa injected as a fake; fetch uses an injected `get_html`). Plotly is the
only new runtime dep, isolated to a `[site]` extra and imported only inside
`site.py`; the heavy `[infer]` (torch/transformers) is needed only for a local scoring run,
never for `--rebuild-only` or CI. `scripts/action_tone_monitor.py` is **superseded** by the
engine (its action measure becomes the `action` column); retire it or leave a thin alias.

## 12. Success criteria (Definition of Done)

1. `pytest` green, offline, including: `upsert_history` idempotency + same-date overwrite;
   `pending_dates` set logic; `tone_deltas` math incl. NaN handling; `redline` opcodes on
   crafted insert/delete/replace/equal pairs; `score_all_measures` merge with an injected
   **fake** RoBERTa classifier + the real lexicons (no torch import); figure builders return
   the expected Plotly trace count on a synthetic history.
2. `python -m cbp.monitor --date 2026-06-17` end-to-end: updates `tone_history.csv` and
   writes a self-contained `index.html` that opens offline with all four panels.
3. `--rebuild-only` regenerates the HTML from the committed CSV with **no** torch installed.
4. `--no-roberta` completes torch-free; the page renders with the RoBERTa line gapped.
5. `--backfill` builds the full 1999→latest history from the calendar.
6. The CI workflow publishes to `gh-pages`; the live Pages URL shows the dashboard; `docs/`
   is absent from the public site.
7. The descriptive-monitor banner + verdict link are present.

## 13. Caveats / risks

- **Not predictive — by construction.** All three measures are descriptive; the action
  measure mirrors the decision, the lexicon is a transparent word-count (regime-confounded
  at h=1), RoBERTa added no OOS value. The banner must say so; no delta should be framed as
  a forecast.
- **Redline is textual, not semantic:** "less restrictive" / "removing accommodation" show
  as reworded text, not interpreted intent (same negation floor as the Phase 2a lexicon).
- **Measure coverage gaps are real, not bugs:** the hawk/dove lexicon is silent on 2024+
  statements and the action measure is 3-level — RoBERTa is included precisely to carry a
  continuous tone on recent statements; show the gaps honestly.
- **RoBERTa license (CC BY-NC 4.0):** the dashboard publishes derived *scores*, not weights,
  for non-commercial research — acceptable; note it. Commercial deploy still blocked.
- **Manual trigger:** if the user forgets to run after a meeting, the page is stale — the
  accepted cost of the manual path (no scheduler by choice). The calendar makes catch-up a
  one-command `--backfill`.
- **First/backfill run is heavy:** scoring ~217 historical statements through RoBERTa once;
  the HTML cache + committed CSV make every subsequent run and all CI rebuilds cheap.
- **`roberta_stance` per-sentence-mean is crude** (Phase 1 caveat) — fine for a descriptive
  series, not to be over-read.
