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
