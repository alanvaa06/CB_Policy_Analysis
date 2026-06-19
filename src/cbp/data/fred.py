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
        # FRED dates are tz-naive calendar dates; localize to UTC so the
        # MarketSeries index is consistently tz-aware (matches release_ts).
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
        return df.sort_index()
