# 001 — FOMC Stance Eval Harness (Phase 0)

**Status:** approved (design) · **Date:** 2026-06-18 · **Owner:** Alan
**Track:** live real-time signal, research-grade (decision 1a/2a)

## 1. Problem & goal

Every academic result linking FOMC text tone to rates is *in-sample* and modest
(Lucca-Trebbi ~30bp, IMF ~20bp). No common-corpus, **out-of-sample** FOMC
benchmark exists — that gap is the project's innovation white-space and its
biggest risk. Before investing in fetch/classify infrastructure, build the
**evaluation harness first** and prove (or disprove) that a text-derived stance
signal has out-of-sample predictive power for rates, with leak-proof alignment.

**Phase 0 goal:** a tested, look-ahead-safe walk-forward harness that scores any
release-aligned stance series against market/policy-rate targets, exercised
end-to-end by a *throwaway* baseline stance signal (no fetcher, no model).

Non-goal for Phase 0: producing a tradeable signal. That is Phase 1 (end-to-end
thin slice), which drops a real FOMC-RoBERTa stance series into this harness.

## 2. Scope

**In scope (Phase 0):**
- Pull `DGS2` (2y Treasury, primary target) and `EFFR` (effective fed funds,
  secondary) from FRED.
- Load FOMC release dates/timestamps and a throwaway stance series from the
  released Trillion Dollar Words (TDW) dataset (1996–2022).
- Align stance@release → forward target change, strictly look-ahead-safe.
- Expanding-window out-of-sample evaluation + event-study diagnostic.
- Metrics, report, CLI entrypoint, full pytest suite incl. a leak test.

**Out of scope (Phase 1+, YAGNI):** real-time fetcher, FOMC-RoBERTa inference,
full / multi-doc-type corpus, CME FedWatch / futures-implied probabilities,
trading-PnL backtest.

## 3. Architecture

`src/` layout, typed throughout, one purpose per module.

```
src/cbp/
  data/  fred.py          # FredClient -> DGS2, EFFR tidy frame
         fomc_calendar.py # FOMC release dates + timestamps
         stance.py        # load throwaway stance series (TDW 1996-2022)
  align/ aligner.py       # stance@release -> forward target, look-ahead-safe
  eval/  walkforward.py   # expanding-window OOS loop
         metrics.py       # OOS R2, RMSE, hit-rate, sign-test
         eventstudy.py    # [t-1, t+1] window diagnostic
  models/baseline.py      # naive (random-walk / mean) + simple regressor
  io/    store.py         # parquet / sqlite read+write
  config.py               # frozen dataclass: horizons, windows, paths
  cli.py                  # run harness -> report
tests/                    # per-module + leak test + synthetic fixtures
```

Each unit must answer: what it does, how to call it, what it depends on. Units
are independently testable through well-defined interfaces.

## 4. Data contracts

- **MarketSeries** — date-indexed DataFrame, columns `DGS2`, `EFFR`. Business-day
  index. Raw NaNs preserved (holiday/gap handling happens at alignment, not here).
- **StanceSeries** — DataFrame `[release_date, release_ts, stance, doc_type]`.
  `stance` is the TDW-derived score (throwaway input for Phase 0).
- **AlignedPanel** — DataFrame `[release_ts, stance, target_{series,h}]` where
  `target = Δ(target_series) over (t, t+h]` business days — **strictly after**
  `release_ts`. One column per (target_series, horizon) pair. Target series:
  `DGS2` (primary), `EFFR` (secondary). Horizons `h ∈ {1, 5, 22}` business days
  (≈ 1 day / 1 week / 1 month), configurable in `config.py`.

## 5. OOS protocol (leak-proof by construction)

Expanding window; events ordered by `release_ts`. For each release *t* after a
minimum train size N₀ (default 20 releases):
1. Train only on releases whose **entire target window has closed before *t***.
2. Predict `target_h` at *t* from `stance@t`; store prediction vs realized.
3. Compute metrics on the OOS prediction set only.
4. The signal must **beat a random-walk / zero-change baseline** OOS to count.

**Look-ahead guards:**
- `target_h` at release *t* uses `DGS2` from `(t, t+h]` — never on or before *t*.
- Training excludes any release whose target window has not fully closed before *t*.
- `DGS2`/`EFFR` are not revised, so data-vintage / real-time-vintage bias does not
  apply here (recorded in §8 caveats); revisit if a revised series is ever added.

## 6. Metrics & report

- OOS R² and RMSE vs the naive baseline.
- Directional hit-rate + sign-test (does hawkish text → higher forward 2y yield).
- Event-study diagnostic: close-to-close Δ in the `[t-1, t+1]` window regressed on
  stance — a free contemporaneous-impact read.
- Report emitted per horizon `h`; CLI prints a summary table + writes artifacts.

## 7. Error handling / edge cases

- FRED holiday/gap inside a target window → **drop that release, log the reason**.
  Never impute the target.
- Release timing: FOMC statements release ~2:00pm ET. If only the date is known,
  use a documented close-to-close convention (Δ measured from release-day close).
- Unscheduled / inter-meeting actions: included but flagged in output.
- Missing stance for a release, or train window < N₀ → skip with a logged reason.

## 8. Testing (TDD, pytest)

- **Synthetic-signal fixture:** inject a known stance→yield relationship; the
  harness must recover it (positive OOS skill).
- **Null fixture:** random/shuffled stance; the harness must report ~zero OOS
  skill — guards against false positives.
- **Look-ahead-leak test (critical):** perturb only future data; train-time
  predictions must not change.
- Unit tests: FRED parsing, alignment join (assert no row references a date ≤
  `release_ts`), metrics math, walk-forward window boundaries.
- Optional hypothesis property test: alignment never references target dates
  on/before `release_ts`.

## 9. Stack & conventions

Python 3.11+; deps via `pyproject.toml` (`pandas`, `numpy`, `statsmodels` or
`scikit-learn`, `fredapi`, `pytest`, `hypothesis`). Storage = parquet files +
a small sqlite index. Default horizons `h ∈ {1, 5, 22}` business days; min train
N₀ = 20 releases (all in `config.py`). Typed code, tests accompany every module
(see `docs/references/python_best_practices.md`). FRED API key via env var.

## 10. Success criteria (Definition of Done, Phase 0)

1. `pytest` green, including the look-ahead-leak test and both synthetic fixtures.
2. CLI runs end-to-end on FRED + TDW data and emits an OOS report per horizon.
3. The harness recovers the injected synthetic signal and rejects the null.
4. A documented verdict on whether the TDW throwaway stance shows any OOS skill
   vs baseline on `DGS2` (the go/no-go read for Phase 1).

## 11. Caveats / risks

- Throwaway stance (TDW) ends 2022 — fine for historical backtest; live fetch is
  Phase 1.
- All targets are final (non-revised) series; acceptable for research v1.
- A null OOS result is a *valid, valuable* outcome — it would redirect Phase 1
  rather than fail the project.
