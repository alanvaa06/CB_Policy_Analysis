# src/cbp/monitor/calendar.py
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_calendar(path: Path) -> list[dt.date]:
    """Load scheduled FOMC announcement dates from a CSV with a `date` column.
    Returns sorted unique `datetime.date`s. Missing file -> ValueError naming the path."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"FOMC calendar not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return sorted({d.date() for d in df["date"]})


def pending_dates(calendar: list[dt.date], history: pd.DataFrame) -> list[dt.date]:
    """Calendar dates with no row yet in `history`. Sorted ascending."""
    scored = set()
    if len(history):
        scored = {d.date() for d in pd.to_datetime(history["date"])}
    return sorted(d for d in calendar if d not in scored)
