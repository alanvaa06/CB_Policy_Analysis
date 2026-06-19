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
        dup_mask = ef["date"].duplicated(keep=False)
        if dup_mask.any():
            dups = sorted({d.date() for d in ef.loc[dup_mask, "date"]})
            raise ValueError(
                "extra_features has duplicate date(s), so per-release lookup is "
                f"ambiguous: {', '.join(str(d) for d in dups)}"
            )
        feat_cols = [c for c in ef.columns if c != "date"]
        feat_lookup = ef.set_index("date")

    # A target series absent from the market frame is a GLOBAL precondition
    # (the same frame is shared by every release), not a per-release window gap.
    # Fail fast naming the missing series rather than dropping every release and
    # silently emptying the whole panel.
    missing = [sid for sid in config.target_series if sid not in market.columns]
    if missing:
        raise ValueError(
            f"target series absent from market frame: {', '.join(missing)}"
        )

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
