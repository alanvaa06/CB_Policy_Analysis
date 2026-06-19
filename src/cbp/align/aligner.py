# src/cbp/align/aligner.py
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from cbp.config import Config

logger = logging.getLogger(__name__)

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
    # release_ts is tz-aware UTC; real FRED data arrives tz-naive. Normalize a
    # tz-naive market index to UTC so the index/ts comparisons in forward_change
    # are valid (a naive calendar date is treated as that date at 00:00 UTC).
    if isinstance(market.index, pd.DatetimeIndex) and market.index.tz is None:
        market = market.copy()
        market.index = market.index.tz_localize("UTC")
    rows = []
    for _, r in stance.sort_values("release_ts").iterrows():
        row = {"release_ts": r["release_ts"], "stance": r["stance"]}
        ok = True
        reasons: list[str] = []
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
