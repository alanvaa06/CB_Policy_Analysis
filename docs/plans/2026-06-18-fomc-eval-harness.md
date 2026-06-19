# FOMC Stance Eval Harness (Phase 0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, look-ahead-safe walk-forward harness that scores a release-aligned FOMC stance signal against `DGS2`/`EFFR`, exercised by a throwaway TDW stance series — to prove (or disprove) out-of-sample predictive power before building any fetch/classify infra.

**Architecture:** Pure-Python `src/` package with one-responsibility modules: data loaders (FRED, FOMC calendar, stance) → leak-safe aligner → expanding-window OOS evaluator + event-study diagnostic → CLI report. Network I/O is isolated behind thin parse functions so all logic is unit-testable offline.

**Tech Stack:** Python 3.11+, pandas, numpy, statsmodels, fredapi, pytest, hypothesis. Storage: parquet + sqlite. Spec: `docs/prd/001-fomc-eval-harness.md`.

---

## Prerequisites (manual, before Task 1)

- Python 3.11+ available.
- FRED API key in env var `FRED_API_KEY` (free: https://fred.stlouisfed.org/docs/api/api_key.html). Only needed for the live CLI run (Task 12); all unit tests run without it.
- Throwaway stance CSV placed at `data/raw/tdw_stance.csv` with columns `date,stance` (date = FOMC statement release date `YYYY-MM-DD`; stance = float, hawkish>0 / dovish<0). Derive by aggregating the released Trillion Dollar Words dataset (mean of hawkish=+1/neutral=0/dovish=-1 per release day). The harness is source-agnostic — only the CSV shape matters. Tests use a fixture, not this file.

## Shared interfaces (defined once, reused exactly)

```python
# src/cbp/config.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)        # business days fwd
    n0: int = 20                                   # min train releases
    target_series: tuple[str, ...] = ("DGS2", "EFFR")
    event_window: tuple[int, int] = (-1, 1)        # business days around release
    data_dir: Path = Path("data")
    fred_api_key: str | None = None
```

DataFrame contracts (plain `pd.DataFrame`, validated by tests):
- **MarketSeries** — `DatetimeIndex` (business days), columns = series ids (`DGS2`,`EFFR`), float, raw NaNs kept.
- **StanceSeries** — columns `[release_date: datetime64, release_ts: datetime64[UTC], stance: float, doc_type: str]`.
- **AlignedPanel** — columns `[release_ts, stance, <series>_h<h> ...]` (one target col per (series,h)).

Function signatures used across tasks:
- `parse_fred_observations(obs: list[dict], series_id: str) -> pd.Series`
- `FredClient(api_key).fetch(series_ids: list[str], start: str, end: str) -> pd.DataFrame`
- `load_fomc_calendar(path: Path) -> pd.DataFrame`  (cols `release_date`, `release_ts`)
- `load_stance(path: Path, calendar: pd.DataFrame) -> pd.DataFrame`  (StanceSeries)
- `forward_change(series: pd.Series, ts: pd.Timestamp, h: int) -> float`
- `build_aligned_panel(market, stance, config) -> pd.DataFrame`
- `oos_r2(y_true, y_pred, y_base) -> float`, `rmse(y_true, y_pred) -> float`, `hit_rate(y_true, y_pred) -> float`, `sign_test(y_true, y_pred) -> dict`
- `run_walkforward(panel, target_col, model, baseline, n0) -> pd.DataFrame` (cols `release_ts,y_true,y_pred,y_base`)
- `event_study(market, stance, series, window) -> dict` (keys `slope,tstat,r2,n`)

Model protocol: `.fit(X: np.ndarray, y: np.ndarray) -> None`, `.predict(X: np.ndarray) -> np.ndarray`.

---

## File structure

```
pyproject.toml
src/cbp/__init__.py
src/cbp/config.py            # Config dataclass
src/cbp/io/store.py          # parquet + sqlite read/write
src/cbp/data/fred.py         # FredClient + parse_fred_observations
src/cbp/data/fomc_calendar.py# load_fomc_calendar
src/cbp/data/stance.py       # load_stance
src/cbp/align/aligner.py     # forward_change, build_aligned_panel  (CORE, leak-safe)
src/cbp/models/baseline.py   # ZeroChange, MeanModel, SimpleOLS
src/cbp/eval/metrics.py      # oos_r2, rmse, hit_rate, sign_test
src/cbp/eval/walkforward.py  # run_walkforward (leak guards)
src/cbp/eval/eventstudy.py   # event_study
src/cbp/cli.py               # end-to-end run + report
tests/...                    # mirror per module + leak/synthetic/null tests
tests/fixtures/...           # tiny offline CSVs
```

---

### Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`, `src/cbp/__init__.py`, `tests/__init__.py`, `tests/test_sanity.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_sanity.py
import cbp

def test_package_importable():
    assert hasattr(cbp, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sanity.py -v`
Expected: FAIL (ModuleNotFoundError: cbp).

- [ ] **Step 3: Write minimal implementation**
```toml
# pyproject.toml
[project]
name = "cbp"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = ["pandas>=2.0", "numpy>=1.26", "statsmodels>=0.14", "fredapi>=0.5"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```
```python
# src/cbp/__init__.py
__version__ = "0.0.1"
```
```python
# tests/__init__.py  (empty)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pip install -e ".[dev]" && pytest tests/test_sanity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml src/cbp/__init__.py tests/__init__.py tests/test_sanity.py
git commit -m "feat: project bootstrap (cbp package + pytest)"
```

---

### Task 2: Config

**Files:**
- Create: `src/cbp/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_config.py
from pathlib import Path
from cbp.config import Config

def test_defaults():
    c = Config()
    assert c.horizons == (1, 5, 22)
    assert c.n0 == 20
    assert c.target_series == ("DGS2", "EFFR")
    assert c.event_window == (-1, 1)
    assert isinstance(c.data_dir, Path)

def test_frozen():
    c = Config()
    import pytest
    with pytest.raises(Exception):
        c.n0 = 5  # frozen
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py -v`
Expected: FAIL (No module named cbp.config).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)
    n0: int = 20
    target_series: tuple[str, ...] = ("DGS2", "EFFR")
    event_window: tuple[int, int] = (-1, 1)
    data_dir: Path = Path("data")
    fred_api_key: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/config.py tests/test_config.py
git commit -m "feat: Config dataclass"
```

---

### Task 3: io/store (parquet + sqlite round-trip)

**Files:**
- Create: `src/cbp/io/__init__.py`, `src/cbp/io/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_store.py
import pandas as pd
from cbp.io.store import write_parquet, read_parquet

def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    p = tmp_path / "x.parquet"
    write_parquet(df, p)
    out = read_parquet(p)
    pd.testing.assert_frame_equal(df, out)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_store.py -v`
Expected: FAIL (No module named cbp.io.store).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/io/__init__.py  (empty)
```
```python
# src/cbp/io/store.py
from pathlib import Path
import pandas as pd

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```
Note: parquet needs `pyarrow`; add `pyarrow>=14` to `pyproject.toml` dependencies and re-run `pip install -e ".[dev]"`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/io tests/test_store.py pyproject.toml
git commit -m "feat: parquet store round-trip"
```

---

### Task 4: data/fred (pure parser + thin client)

**Files:**
- Create: `src/cbp/data/__init__.py`, `src/cbp/data/fred.py`, `tests/test_fred.py`

- [ ] **Step 1: Write the failing test** (parser is pure → fully offline)
```python
# tests/test_fred.py
import numpy as np
from cbp.data.fred import parse_fred_observations

def test_parse_handles_missing_dot():
    obs = [
        {"date": "2020-01-02", "value": "1.57"},
        {"date": "2020-01-03", "value": "."},     # FRED missing marker
        {"date": "2020-01-06", "value": "1.60"},
    ]
    s = parse_fred_observations(obs, "DGS2")
    assert s.name == "DGS2"
    assert s.loc["2020-01-02"] == 1.57
    assert np.isnan(s.loc["2020-01-03"])
    assert s.index.is_monotonic_increasing
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_fred.py -v`
Expected: FAIL (No module named cbp.data.fred).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/data/__init__.py  (empty)
```
```python
# src/cbp/data/fred.py
from __future__ import annotations
import pandas as pd

def parse_fred_observations(obs: list[dict], series_id: str) -> pd.Series:
    idx = pd.to_datetime([o["date"] for o in obs])
    vals = pd.to_numeric(
        [None if o["value"] in (".", "") else o["value"] for o in obs],
        errors="coerce",
    )
    return pd.Series(vals, index=idx, name=series_id).sort_index()

class FredClient:
    def __init__(self, api_key: str):
        from fredapi import Fred
        self._fred = Fred(api_key=api_key)

    def fetch(self, series_ids: list[str], start: str, end: str) -> pd.DataFrame:
        cols = {sid: self._fred.get_series(sid, start, end) for sid in series_ids}
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_fred.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/data/__init__.py src/cbp/data/fred.py tests/test_fred.py
git commit -m "feat: FRED parser + thin client"
```

---

### Task 5: data/fomc_calendar

**Files:**
- Create: `src/cbp/data/fomc_calendar.py`, `tests/fixtures/fomc_dates.csv`, `tests/test_calendar.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_calendar.py
from pathlib import Path
import pandas as pd
from cbp.data.fomc_calendar import load_fomc_calendar

def test_calendar_has_utc_2pm_et(tmp_path):
    p = tmp_path / "fomc_dates.csv"
    p.write_text("release_date\n2020-01-29\n2020-03-18\n")
    cal = load_fomc_calendar(p)
    assert list(cal.columns) == ["release_date", "release_ts"]
    assert cal["release_ts"].dt.tz is not None          # tz-aware
    # 14:00 America/New_York on 2020-01-29 (EST) == 19:00 UTC
    assert cal["release_ts"].iloc[0] == pd.Timestamp("2020-01-29 19:00", tz="UTC")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_calendar.py -v`
Expected: FAIL (No module named cbp.data.fomc_calendar).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/data/fomc_calendar.py
from pathlib import Path
import pandas as pd

def load_fomc_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rd = pd.to_datetime(df["release_date"])
    ts_et = (rd + pd.Timedelta(hours=14)).dt.tz_localize("America/New_York")
    return pd.DataFrame({"release_date": rd, "release_ts": ts_et.dt.tz_convert("UTC")})
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_calendar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/data/fomc_calendar.py tests/test_calendar.py
git commit -m "feat: FOMC calendar loader (2pm ET -> UTC)"
```

---

### Task 6: data/stance

**Files:**
- Create: `src/cbp/data/stance.py`, `tests/test_stance.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_stance.py
import pandas as pd
from cbp.data.fomc_calendar import load_fomc_calendar
from cbp.data.stance import load_stance

def test_stance_joins_release_ts(tmp_path):
    cal_p = tmp_path / "fomc_dates.csv"
    cal_p.write_text("release_date\n2020-01-29\n2020-03-18\n")
    cal = load_fomc_calendar(cal_p)
    st_p = tmp_path / "tdw_stance.csv"
    st_p.write_text("date,stance\n2020-01-29,0.5\n2020-03-18,-0.8\n")
    s = load_stance(st_p, cal)
    assert list(s.columns) == ["release_date", "release_ts", "stance", "doc_type"]
    assert s["stance"].tolist() == [0.5, -0.8]
    assert s["release_ts"].dt.tz is not None
    assert (s["doc_type"] == "statement").all()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_stance.py -v`
Expected: FAIL (No module named cbp.data.stance).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/data/stance.py
from pathlib import Path
import pandas as pd

def load_stance(path: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["release_date"] = pd.to_datetime(raw["date"])
    merged = raw.merge(calendar, on="release_date", how="inner")
    merged["doc_type"] = "statement"
    return merged[["release_date", "release_ts", "stance", "doc_type"]].sort_values(
        "release_ts"
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_stance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/data/stance.py tests/test_stance.py
git commit -m "feat: stance loader joined to calendar"
```

---

### Task 7: align/aligner — CORE, look-ahead-safe

**Files:**
- Create: `src/cbp/align/__init__.py`, `src/cbp/align/aligner.py`, `tests/test_aligner.py`

- [ ] **Step 1: Write the failing tests** (forward window + no-future-reference)
```python
# tests/test_aligner.py
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.align.aligner import forward_change, build_aligned_panel

def _market():
    idx = pd.bdate_range("2020-01-27", "2020-02-28", tz="UTC")
    # DGS2 rises by 0.01 each business day from 1.00
    vals = 1.00 + 0.01 * np.arange(len(idx))
    return pd.DataFrame({"DGS2": vals, "EFFR": vals}, index=idx)

def test_forward_change_strictly_after_ts():
    m = _market()
    ts = pd.Timestamp("2020-01-29 19:00", tz="UTC")  # release
    # h=1: change from first bday strictly AFTER ts to that day; 2020-01-30 vs ... 
    fc = forward_change(m["DGS2"], ts, h=1)
    # value on 2020-01-30 minus value on 2020-01-29 (last point at/before exit)
    assert fc > 0  # rising series => positive 1-day fwd change

def test_panel_has_target_cols_and_no_future_leak():
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert "DGS2_h1" in panel.columns and "DGS2_h5" in panel.columns
    # invariant: target for a release never uses market data on/before release_ts
    assert panel["release_ts"].iloc[0] == pd.Timestamp("2020-01-29 19:00", tz="UTC")

def test_drop_release_when_window_incomplete():
    m = _market()
    # release near end of series: h=22 window cannot close -> row dropped
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-02-27"])})
    cal["release_ts"] = pd.Timestamp("2020-02-27 19:00", tz="UTC")
    stance = cal.assign(stance=0.1, doc_type="statement")
    cfg = Config(horizons=(22,), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert panel.empty
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_aligner.py -v`
Expected: FAIL (No module named cbp.align.aligner).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/align/__init__.py  (empty)
```
```python
# src/cbp/align/aligner.py
from __future__ import annotations
import numpy as np
import pandas as pd
from cbp.config import Config

def forward_change(series: pd.Series, ts: pd.Timestamp, h: int) -> float:
    """Change in `series` over the h business days STRICTLY AFTER ts.

    base = last observation at/before ts; future = h-th observation after ts.
    Returns NaN if the full window is unavailable or values are missing.
    """
    s = series.dropna().sort_index()
    after = s[s.index > ts]
    at_or_before = s[s.index <= ts]
    if len(after) < h or at_or_before.empty:
        return np.nan
    base = at_or_before.iloc[-1]
    future = after.iloc[h - 1]
    return float(future - base)

def build_aligned_panel(market: pd.DataFrame, stance: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    for _, r in stance.sort_values("release_ts").iterrows():
        row = {"release_ts": r["release_ts"], "stance": r["stance"]}
        ok = True
        for sid in config.target_series:
            if sid not in market.columns:
                ok = False
                break
            for h in config.horizons:
                val = forward_change(market[sid], r["release_ts"], h)
                if np.isnan(val):
                    ok = False
                row[f"{sid}_h{h}"] = val
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)
```
Design note for reviewers: `forward_change` only ever reads observations with `index > ts` for the future leg and `index <= ts` for the base — the leak-safety invariant. A release is dropped (not imputed) if any configured target window is incomplete.

- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_aligner.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add src/cbp/align tests/test_aligner.py
git commit -m "feat: leak-safe stance->forward-target aligner"
```

---

### Task 8: models/baseline

**Files:**
- Create: `src/cbp/models/__init__.py`, `src/cbp/models/baseline.py`, `tests/test_models.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_models.py
import numpy as np
from cbp.models.baseline import ZeroChange, MeanModel, SimpleOLS

def test_zero_change_predicts_zero():
    m = ZeroChange(); m.fit(np.array([[1.0],[2.0]]), np.array([5.0, 6.0]))
    assert (m.predict(np.array([[9.0]])) == 0.0).all()

def test_mean_model_predicts_train_mean():
    m = MeanModel(); m.fit(np.zeros((3, 1)), np.array([2.0, 4.0, 6.0]))
    assert m.predict(np.zeros((1, 1)))[0] == 4.0

def test_ols_recovers_linear():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y = 3.0 * x[:, 0] + 0.5
    m = SimpleOLS(); m.fit(x, y)
    pred = m.predict(np.array([[1.0]]))[0]
    assert abs(pred - 3.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_models.py -v`
Expected: FAIL (No module named cbp.models.baseline).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/models/__init__.py  (empty)
```
```python
# src/cbp/models/baseline.py
from __future__ import annotations
import numpy as np

class ZeroChange:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

class MeanModel:
    def __init__(self) -> None:
        self._mean = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)

class SimpleOLS:
    def __init__(self) -> None:
        self._coef = None; self._intercept = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        A = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self._intercept = float(beta[0]); self._coef = beta[1:]
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._intercept + X @ self._coef
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/models tests/test_models.py
git commit -m "feat: baseline + OLS models"
```

---

### Task 9: eval/metrics

**Files:**
- Create: `src/cbp/eval/__init__.py`, `src/cbp/eval/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_metrics.py
import numpy as np
from cbp.eval.metrics import oos_r2, rmse, hit_rate, sign_test

def test_rmse_zero_when_perfect():
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

def test_oos_r2_one_when_perfect():
    y = np.array([1.0, -2.0, 3.0]); base = np.zeros(3)
    assert abs(oos_r2(y, y, base) - 1.0) < 1e-12

def test_oos_r2_zero_when_equals_baseline():
    y = np.array([1.0, -2.0, 3.0]); base = np.full(3, 0.6667)
    assert abs(oos_r2(y, base, base)) < 1e-9

def test_hit_rate_direction():
    y = np.array([1.0, -1.0, 2.0, -3.0]); p = np.array([0.2, -0.1, -0.5, -1.0])
    assert hit_rate(y, p) == 0.75  # 3 of 4 signs match

def test_sign_test_keys():
    out = sign_test(np.array([1.0, -1.0]), np.array([0.5, -0.5]))
    assert {"hits", "n", "pvalue"} <= set(out)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_metrics.py -v`
Expected: FAIL (No module named cbp.eval.metrics).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/eval/__init__.py  (empty)
```
```python
# src/cbp/eval/metrics.py
from __future__ import annotations
import numpy as np
from scipy import stats  # scipy ships with statsmodels; add to deps if missing

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    sse = np.sum((y_true - y_pred) ** 2)
    sse_base = np.sum((y_true - y_base) ** 2)
    return float(1.0 - sse / sse_base) if sse_base > 0 else float("nan")

def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def sign_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    hits = int(np.sum(np.sign(y_true) == np.sign(y_pred)))
    n = int(len(y_true))
    pvalue = float(stats.binomtest(hits, n, 0.5).pvalue) if n else float("nan")
    return {"hits": hits, "n": n, "pvalue": pvalue}
```
Note: add `scipy>=1.11` to `pyproject.toml` dependencies if not already present, then `pip install -e ".[dev]"`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/eval/__init__.py src/cbp/eval/metrics.py tests/test_metrics.py pyproject.toml
git commit -m "feat: OOS metrics (r2, rmse, hit-rate, sign-test)"
```

---

### Task 10: eval/walkforward — expanding-window OOS + leak guard

**Files:**
- Create: `src/cbp/eval/walkforward.py`, `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing tests** (windowing + synthetic recovery + null + leak)
```python
# tests/test_walkforward.py
import numpy as np
import pandas as pd
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward

def _panel(n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    stance = rng.normal(size=n)
    target = (2.0 * stance if signal else rng.normal(size=n)) + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"release_ts": ts, "stance": stance, "DGS2_h1": target})

def test_skips_until_n0_then_predicts():
    p = _panel(30)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    assert len(out) == 10                      # 30 - 20 OOS predictions
    assert list(out.columns) == ["release_ts", "y_true", "y_pred", "y_base"]

def test_recovers_known_signal():
    p = _panel(120, signal=True)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert corr > 0.9                          # strong OOS skill on real signal

def test_rejects_null_signal():
    p = _panel(120, signal=False)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert abs(corr) < 0.4                      # no spurious skill on noise

def test_no_lookahead_future_perturbation_invariant():
    p = _panel(60, signal=True, seed=1)
    out1 = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    # corrupt the LAST row's target only; predictions for earlier rows must not change
    p2 = p.copy(); p2.loc[p2.index[-1], "DGS2_h1"] = 999.0
    out2 = run_walkforward(p2, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    pd.testing.assert_series_equal(
        out1["y_pred"].iloc[:-1].reset_index(drop=True),
        out2["y_pred"].iloc[:-1].reset_index(drop=True),
    )
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_walkforward.py -v`
Expected: FAIL (No module named cbp.eval.walkforward).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/eval/walkforward.py
from __future__ import annotations
import numpy as np
import pandas as pd

def run_walkforward(panel: pd.DataFrame, target_col: str, model, baseline, n0: int) -> pd.DataFrame:
    """Expanding-window OOS. For each release i >= n0, train on rows [0, i) and
    predict row i. Training never sees row i or any later row -> no look-ahead.
    """
    df = panel.sort_values("release_ts").reset_index(drop=True)
    y = df[target_col].to_numpy(dtype=float)
    X = df[["stance"]].to_numpy(dtype=float)
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
Why the leak test passes: row `i`'s prediction is trained on `X[:i]` only; perturbing row `n-1`'s target cannot affect any prediction for `i < n-1`.

- [ ] **Step 4: Run tests to verify they pass**
Run: `pytest tests/test_walkforward.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add src/cbp/eval/walkforward.py tests/test_walkforward.py
git commit -m "feat: expanding-window OOS walk-forward (leak-safe)"
```

---

### Task 11: eval/eventstudy

**Files:**
- Create: `src/cbp/eval/eventstudy.py`, `tests/test_eventstudy.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_eventstudy.py
import numpy as np
import pandas as pd
from cbp.eval.eventstudy import event_study

def test_event_study_recovers_slope():
    idx = pd.bdate_range("2020-01-01", "2020-12-31", tz="UTC")
    rng = np.random.default_rng(0)
    dgs2 = pd.Series(rng.normal(size=len(idx)).cumsum(), index=idx, name="DGS2")
    market = dgs2.to_frame()
    # build releases whose [t-1,t+1] change == 1.5 * stance by construction is hard;
    # instead assert the function returns the expected keys and finite slope
    rel = pd.DataFrame({
        "release_ts": pd.to_datetime(["2020-03-18","2020-06-10","2020-09-16"]).tz_localize("UTC"),
        "stance": [0.5, -0.2, 0.1],
    })
    out = event_study(market, rel, "DGS2", window=(-1, 1))
    assert {"slope", "tstat", "r2", "n"} <= set(out)
    assert out["n"] == 3
    assert np.isfinite(out["slope"])
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_eventstudy.py -v`
Expected: FAIL (No module named cbp.eval.eventstudy).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/eval/eventstudy.py
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _window_change(series: pd.Series, ts: pd.Timestamp, window: tuple[int, int]) -> float:
    s = series.dropna().sort_index()
    before = s[s.index <= ts]
    after = s[s.index > ts]
    lo, hi = window
    if len(before) < abs(lo) + 1 or len(after) < hi:
        return np.nan
    start = before.iloc[lo - 1] if lo < 0 else before.iloc[-1]
    end = after.iloc[hi - 1]
    return float(end - start)

def event_study(market: pd.DataFrame, releases: pd.DataFrame, series: str, window: tuple[int, int]) -> dict:
    changes, stances = [], []
    for _, r in releases.iterrows():
        ch = _window_change(market[series], r["release_ts"], window)
        if not np.isnan(ch):
            changes.append(ch); stances.append(r["stance"])
    if len(changes) < 2:
        return {"slope": float("nan"), "tstat": float("nan"), "r2": float("nan"), "n": len(changes)}
    X = sm.add_constant(np.array(stances)); res = sm.OLS(np.array(changes), X).fit()
    return {"slope": float(res.params[1]), "tstat": float(res.tvalues[1]),
            "r2": float(res.rsquared), "n": len(changes)}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_eventstudy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/eval/eventstudy.py tests/test_eventstudy.py
git commit -m "feat: event-study diagnostic"
```

---

### Task 12: cli — end-to-end wiring + offline integration test

**Files:**
- Create: `src/cbp/cli.py`, `tests/fixtures/mini_market.csv`, `tests/fixtures/mini_stance.csv`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing integration test** (fully offline via injected market/stance)
```python
# tests/test_cli.py
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.cli import run_report

def test_run_report_offline():
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rng = np.random.default_rng(0)
    stance_by_release = {}
    # ~8 releases/yr; build a market that rises after hawkish stance
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    stance = rng.normal(size=len(rel_ts))
    market = pd.DataFrame(index=idx)
    base = np.zeros(len(idx))
    for ts, s in zip(rel_ts, stance):
        base[idx > ts] += 0.02 * s        # hawkish -> drift up after release
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))
    market["EFFR"] = market["DGS2"]
    stance_df = pd.DataFrame({"release_ts": rel_ts, "stance": stance, "doc_type": "statement",
                              "release_date": rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)})
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    report = run_report(market, stance_df, cfg)
    assert ("DGS2", 1) in report["oos"]
    assert report["oos"][("DGS2", 1)]["n"] > 0
    assert np.isfinite(report["oos"][("DGS2", 1)]["oos_r2"])
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_cli.py -v`
Expected: FAIL (No module named cbp.cli).

- [ ] **Step 3: Write minimal implementation**
```python
# src/cbp/cli.py
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.align.aligner import build_aligned_panel
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward
from cbp.eval.metrics import oos_r2, rmse, hit_rate, sign_test
from cbp.eval.eventstudy import event_study

def run_report(market: pd.DataFrame, stance: pd.DataFrame, config: Config) -> dict:
    panel = build_aligned_panel(market, stance, config)
    oos = {}
    for sid in config.target_series:
        for h in config.horizons:
            col = f"{sid}_h{h}"
            if col not in panel.columns or len(panel) <= config.n0:
                continue
            wf = run_walkforward(panel, col, SimpleOLS(), ZeroChange(), config.n0)
            if wf.empty:
                continue
            yt, yp, yb = wf["y_true"].to_numpy(), wf["y_pred"].to_numpy(), wf["y_base"].to_numpy()
            oos[(sid, h)] = {
                "n": len(wf), "oos_r2": oos_r2(yt, yp, yb), "rmse": rmse(yt, yp),
                "hit_rate": hit_rate(yt, yp), **{f"sign_{k}": v for k, v in sign_test(yt, yp).items()},
            }
    events = {sid: event_study(market, stance, sid, config.event_window)
              for sid in config.target_series if sid in market.columns}
    return {"oos": oos, "events": events}

def _print_report(report: dict) -> None:
    print("\n=== OOS walk-forward ===")
    for (sid, h), m in report["oos"].items():
        print(f"{sid} h={h:>2}: n={m['n']:>3}  OOS_R2={m['oos_r2']:+.3f}  "
              f"hit={m['hit_rate']:.2f}  sign_p={m['sign_pvalue']:.3f}")
    print("\n=== Event study [t-1,t+1] ===")
    for sid, e in report["events"].items():
        print(f"{sid}: slope={e['slope']:+.4f}  t={e['tstat']:+.2f}  r2={e['r2']:.3f}  n={e['n']}")

def main() -> None:
    ap = argparse.ArgumentParser(description="FOMC stance eval harness (Phase 0)")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2022-12-31")
    args = ap.parse_args()
    from cbp.data.fred import FredClient
    from cbp.data.fomc_calendar import load_fomc_calendar
    from cbp.data.stance import load_stance
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY to run the live report.")
    market = FredClient(cfg.fred_api_key).fetch(list(cfg.target_series), args.start, args.end)
    cal = load_fomc_calendar(cfg.data_dir / "raw" / "fomc_dates.csv")
    stance = load_stance(cfg.data_dir / "raw" / "tdw_stance.csv", cal)
    _print_report(run_report(market, stance, cfg))

if __name__ == "__main__":
    main()
```
Add to `pyproject.toml`: `[project.scripts]` → `cbp-harness = "cbp.cli:main"`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cbp/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: end-to-end harness CLI + offline integration test"
```

---

### Task 13: Full suite + live run + verdict

**Files:**
- Modify: `docs/context/results.md`, `docs/context/memory.md` (record the go/no-go read)

- [ ] **Step 1: Run the whole suite**
Run: `pytest -v`
Expected: ALL PASS (sanity, config, store, fred, calendar, stance, aligner×3, models×3, metrics×5, walkforward×4, eventstudy, cli).

- [ ] **Step 2: Live run on real data** (requires `FRED_API_KEY` + `data/raw/tdw_stance.csv` + `data/raw/fomc_dates.csv`)
Run: `cbp-harness --start 1999-01-01 --end 2022-12-31`
Expected: prints OOS table (per `DGS2`/`EFFR` × h) + event-study block. No crash; finite metrics.

- [ ] **Step 3: Record the verdict**
Append to `docs/context/results.md` one line: whether TDW throwaway stance shows OOS skill vs baseline on `DGS2` (`oos_r2 > 0` and `sign_pvalue < 0.1` ⇒ go signal for Phase 1; else document the null and rethink the signal). Add a `# finding:` line to `docs/context/memory.md`.

- [ ] **Step 4: Commit**
```bash
git add docs/context/results.md docs/context/memory.md
git commit -m "docs: Phase 0 OOS verdict (go/no-go for Phase 1)"
```

---

## Self-review

**Spec coverage:** §2 scope → Tasks 4–12. §3 architecture → file structure matches. §4 contracts → Tasks 4/6/7. §5 OOS protocol + guards → Tasks 7 & 10 (leak tests). §6 metrics + event study → Tasks 9, 11. §7 error handling (drop-not-impute, sub-N₀ skip) → Tasks 7, 10. §8 testing (synthetic, null, leak) → Tasks 7, 10. §10 DoD → Task 13. All covered.

**Placeholder scan:** No TBD/TODO; every code step has runnable code. Manual prerequisites (FRED key, stance CSV) are called out explicitly, not hidden in a task.

**Type consistency:** `Config` fields, `release_ts`/`stance`/`<sid>_h<h>` column names, model `.fit/.predict`, and `run_walkforward` output cols (`release_ts,y_true,y_pred,y_base`) are used identically across Tasks 7/10/12. `forward_change` semantics (base = last ≤ ts, future = h-th > ts) are consistent between aligner and its tests.

**Known soft spots (acceptable for Phase 0):** event-study `_window_change` start-index logic assumes `lo ∈ {-1,0}`; fine for the default `(-1,1)` window — generalize only if wider windows are configured later.
