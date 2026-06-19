# 002 — FOMC Statement Stance: Marginal Effect Beyond the Policy Surprise (Phase 1)

**Status:** approved (design) · **Date:** 2026-06-19 · **Owner:** Alan
**Track:** live real-time signal, research-grade · **Builds on:** Phase 0 harness (001), merged to main.

## 1. Problem & goal

Phase 0 built and validated a leak-safe out-of-sample (OOS) eval harness, and the
per-document-type diagnostic taught the core lesson: **the policy DECISION confounds
the policy TEXT** — any tone measure on meeting-day documents co-moves mechanically
with the same-day rate action, so a raw tone→rate result is not evidence the *text*
carries information.

**Phase 1 goal:** answer one question rigorously — **does the stance of the FOMC
post-meeting statement carry predictive information for market rates BEYOND the
known monetary-policy surprise?** Operationally: build a real stance signal from
standalone FOMC statements (1999+) via FOMC-RoBERTa, and test whether adding stance
to a surprise-only model **improves out-of-sample R²** (nested-model comparison).

**Why statements:** densest real-time, single-document-type, same-day high-impact
series; TDW never labeled standalone statements, so this is genuinely new.

Non-goal: a production trading signal. This is the decisive research test that
tells us whether the text signal is worth productionizing.

## 2. Scope

**In scope (Phase 1 v1):**
- Fetch FOMC post-meeting statements (1999+) from federalreserve.gov.
- Score each statement's stance with FOMC-RoBERTa (hawkish/dovish/neutral → per-statement mean).
- Load the Bauer-Swanson **orthogonalized** monetary-policy-surprise series (SF Fed) as the control.
- Targets `DGS2` (2y, primary) and `DGS1` (1y, front-end) from FRED. **EFFR dropped** (mechanical).
- Extend the harness to multi-feature; run **nested OOS**: surprise-only vs surprise+stance.
- Report ΔR² per (target, horizon) + residual event-study cross-check.

**Out of scope (YAGNI):** OIS targets, press conferences, minutes, intraday windows,
model ensembles, TDW-correlation as a hard gate (sanity check only).

## 3. Architecture

Extends `src/cbp` (Phase 0). One responsibility per module; network/inference
isolated behind pure functions so logic is unit-testable offline.

```
NEW:
  data/fomc_statements.py   # parse_statement_html (pure) + fetch_statements (cache)
  data/mp_surprise.py       # load_surprise: BS orthogonalized series -> [date, surprise]
  models/stance_scorer.py   # score_statements: sentences -> hawk/dove/neutral -> mean
  eval/nested.py            # nested_oos: surprise-only vs surprise+stance OOS R2
EXTENDED:
  align/aligner.py          # build_aligned_panel gains extra_features (surprise) join
  eval/walkforward.py       # run_walkforward(..., feature_cols=...) (SimpleOLS already multivariate)
  config.py                 # targets (DGS2,DGS1), roberta model id, paths
  cli.py                    # Phase 1 nested-comparison report
```

## 4. Data contracts

- **Statements** — `DataFrame[date: datetime64, text: str]`, one row per FOMC statement.
- **Stance** — `DataFrame[date: datetime64, stance: float]`, per-statement mean of mapped
  sentence labels (hawkish +1 / dovish −1 / neutral 0), i.e. the TDW `our_measure` method.
- **Surprise** — `DataFrame[date: datetime64, surprise: float]`, BS orthogonalized surprise (bp), one per meeting.
- **AlignedPanel** — `[release_ts, stance, surprise, <sid>_h<h> ...]`; targets strictly
  after `release_ts` (Phase 0 leak-safety unchanged); `surprise` joined on the release date.

Signatures:
- `parse_statement_html(html: str) -> str`
- `fetch_statements(start_year: int, cache_dir: Path) -> pd.DataFrame`
- `score_statements(statements: pd.DataFrame, classifier) -> pd.DataFrame`  (classifier injected)
- `load_fomc_roberta() -> classifier`  (factory: lazy-loads the real FOMC-RoBERTa pipeline; the CLI passes its output to `score_statements`, tests pass a fake)
- `load_surprise(path: Path) -> pd.DataFrame`
- `build_aligned_panel(market, stance, config, extra_features=None) -> pd.DataFrame`
- `run_walkforward(panel, target_col, feature_cols, model, baseline, n0) -> pd.DataFrame`
- `nested_oos(panel, target_col, n0) -> dict`  (keys: `r2_base, r2_full, delta_r2, n, stance_partial_t`)

## 5. Statement fetching

federalreserve.gov has two URL eras:
- Modern (2006+): `/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`
- Historical (1999–2005): `/boarddocs/press/monetary/{YYYY}/{YYYYMMDD}/` (and `/general/` variants).

`fetch_statements` tries the applicable pattern per FOMC date (dates from the Phase 0
calendar / a bundled FOMC meeting list), caches raw HTML under `data/raw/statements/`,
and parses the statement body via `parse_statement_html`. A date that 404s or yields
empty text is **logged and skipped** (never fabricated). Statements are systematic
from May 1999; earlier meetings without a statement are simply absent.

## 6. Stance scoring

`score_statements` splits each statement into sentences (deterministic splitter),
classifies each with the injected `classifier` (FOMC-RoBERTa: `gtfintechlab/FOMC-RoBERTa`,
labels LABEL_0=Dovish→−1, LABEL_1=Hawkish→+1, LABEL_2=Neutral→0), and sets
`stance = mean(mapped labels)` per statement. The real model is loaded lazily via
`transformers` and cached on disk; scores are cached to parquet so re-runs are cheap.
Per-sentence inputs are truncated to the model's max length. **License: CC BY-NC 4.0
— research use only** (consistent with the research-grade track; blocks commercial
deployment without relicensing).

## 7. Eval — nested OOS (the answer)

For each `target ∈ {DGS2, DGS1}` and `h ∈ {1, 5, 22}`:
- `nested_oos` runs the Phase 0 walk-forward twice on the same panel and N₀:
  feature set **A = [surprise]** (control only) and **B = [surprise, stance]**.
- `delta_r2 = oos_r2(B) − oos_r2(A)`. **The statement text carries marginal predictive
  information iff `delta_r2 > 0`** (the text improves OOS fit beyond the known surprise),
  corroborated by `stance_partial_t` — the t-statistic of the stance coefficient from a
  single in-sample OLS `target ~ surprise + stance` over the full aligned panel (a
  descriptive companion to the OOS ΔR², not itself an OOS metric).
- Cross-check: event-study of the residual `target − (surprise-only OOS fit)` regressed
  on stance — a hawkish-text → higher-residual-yield read independent of the decision.

**Definition of "text adds value":** `delta_r2 > 0` on `DGS2` at ≥1 horizon, with a
consistently-signed stance coefficient. A null (`delta_r2 ≤ 0` everywhere) is a valid,
publishable result: statement *tone* (as measured) adds nothing beyond the surprise.

## 8. Error handling / edge cases

- URL-era mismatch / 404 / empty parse → log + skip the date.
- Missing stance or missing surprise for a release → drop that release + log (reuse Phase 0 logging).
- FRED holidays inside a target window → drop release (Phase 0 behavior).
- Unscheduled / inter-meeting actions → included but flagged.
- Long statements → per-sentence truncation to model max tokens.
- `DGS1`/`DGS2` are non-revised → no vintage bias (recorded caveat).

## 9. Stack & conventions

Python 3.11+ (runs on 3.14). Adds: `torch` (CPU), `transformers`, `huggingface_hub`,
`beautifulsoup4`, `requests`, a sentence splitter (`blingfire` or a tested regex).
Reuse Phase 0 deps. Typed code; tests accompany every module. RoBERTa-large (~1.4GB)
downloaded once and cached; CPU inference over ~190 statements is minutes.

## 10. Success criteria (Definition of Done)

1. `pytest` green, including: HTML→text (both eras), fake-classifier→mean stance,
   fixture→surprise, multi-feature aligner preserves the no-leak invariant, and a
   **synthetic test where stance adds known signal → ΔR²>0** plus a **null where it
   does not → ΔR²≈0**.
2. CLI runs end-to-end (FRED + statements + RoBERTa + BS surprise) and prints the
   nested OOS table (ΔR² per target × horizon) + residual event-study.
3. A documented verdict: does statement stance add OOS predictive value beyond the
   BS surprise on `DGS2`? (go/no-go for productionizing the text signal.)

## 11. Caveats / risks

- **BS file coverage:** confirm the SF Fed `monetary-policy-surprises-data.xlsx`
  reaches back to 1999; if it is truncated to recent years, use the SF Fed USMPD
  database or the Bauer-Swanson (2023) replication set (1988–2019) for the 1999+ window.
- **Sentence-stance ≠ market-calibrated stance:** the per-sentence mean is a crude
  aggregate; a null result may reflect the aggregation, not the absence of information.
  Record this when interpreting.
- **Statement scraping fragility:** federalreserve.gov layout differs across years;
  the parser must be fixture-tested on both eras and skip-on-failure, not guess.
- **CC BY-NC** blocks commercial deployment of the FOMC-RoBERTa-derived signal.
- All targets final/non-revised; in-sample-vs-OOS handled by the walk-forward, but
  N₀=20 with ~190 statements leaves a modest OOS sample — magnitudes will be noisy.
