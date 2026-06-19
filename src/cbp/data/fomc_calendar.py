# src/cbp/data/fomc_calendar.py
from pathlib import Path
import pandas as pd

def load_fomc_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rd = pd.to_datetime(df["release_date"])
    ts_et = (rd + pd.Timedelta(hours=14)).dt.tz_localize("America/New_York")
    return pd.DataFrame({"release_date": rd, "release_ts": ts_et.dt.tz_convert("UTC")})
