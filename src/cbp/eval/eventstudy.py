# src/cbp/eval/eventstudy.py
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _window_change(series: pd.Series, ts: pd.Timestamp, window: tuple[int, int]) -> float:
    lo, hi = window
    assert lo <= 0, f"window lo must be <= 0 (anchored at/before event), got {lo}"
    s = series.dropna().sort_index()
    before = s[s.index <= ts]
    after = s[s.index > ts]
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
    if len(changes) < 2 or len(set(stances)) < 2:
        return {"slope": float("nan"), "tstat": float("nan"), "r2": float("nan"), "n": len(changes)}
    X = sm.add_constant(np.array(stances)); res = sm.OLS(np.array(changes), X).fit()
    return {"slope": float(res.params[1]), "tstat": float(res.tvalues[1]),
            "r2": float(res.rsquared), "n": len(changes)}
