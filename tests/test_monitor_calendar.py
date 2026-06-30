# tests/test_monitor_calendar.py
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from cbp.monitor.calendar import load_calendar, pending_dates


def test_load_calendar_parses_sorted_unique(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date\n2024-03-20\n2024-01-31\n2024-01-31\n")
    assert load_calendar(p) == [dt.date(2024, 1, 31), dt.date(2024, 3, 20)]


def test_load_calendar_missing_raises_valueerror():
    with pytest.raises(ValueError, match="calendar"):
        load_calendar("does/not/exist.csv")


def test_pending_dates_excludes_already_scored():
    cal = [dt.date(2024, 1, 31), dt.date(2024, 3, 20), dt.date(2024, 5, 1)]
    history = pd.DataFrame({"date": pd.to_datetime(["2024-01-31"])})
    assert pending_dates(cal, history) == [dt.date(2024, 3, 20), dt.date(2024, 5, 1)]


def test_pending_dates_empty_history_returns_all():
    cal = [dt.date(2024, 1, 31), dt.date(2024, 3, 20)]
    history = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    assert pending_dates(cal, history) == cal


def test_repo_calendar_exists_and_covers_recent_meetings():
    cal = load_calendar(Path("data/monitor/fomc_calendar.csv"))
    assert len(cal) >= 180                      # 1999→2026 ≈ 8/yr
    assert dt.date(2026, 6, 17) in cal          # last RECENT_DATES entry
    assert min(cal).year <= 1999 and max(cal).year >= 2026
