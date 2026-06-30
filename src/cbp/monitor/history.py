# src/cbp/monitor/history.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = ["date", "action", "lexicon_tone", "roberta_stance", "n_sentences"]


def load_history(path: Path) -> pd.DataFrame:
    """Load the committed tone history. Missing file -> empty frame with the schema
    (date as datetime64). Always returns exactly HISTORY_COLUMNS, sorted by date."""
    path = Path(path)
    if not path.exists():
        empty = {c: pd.Series(dtype="datetime64[ns]" if c == "date" else "float64")
                 for c in HISTORY_COLUMNS}
        return pd.DataFrame(empty)
    df = pd.read_csv(path, parse_dates=["date"])
    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[HISTORY_COLUMNS].sort_values("date").reset_index(drop=True)


def upsert_history(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append `new` rows; on a duplicate date the new row wins. Result is sorted by
    date, de-duplicated, index reset. Idempotent: re-upserting identical rows is a no-op."""
    combined = pd.concat([history, new[HISTORY_COLUMNS]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return (combined.drop_duplicates("date", keep="last")
                    .sort_values("date").reset_index(drop=True))


def save_history(history: pd.DataFrame, path: Path) -> None:
    """Write the history to CSV with date as YYYY-MM-DD (stable, diff-friendly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = history.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
