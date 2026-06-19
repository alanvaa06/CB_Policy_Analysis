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
