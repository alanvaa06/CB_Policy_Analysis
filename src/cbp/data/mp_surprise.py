# src/cbp/data/mp_surprise.py
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_surprise(
    path: Path,
    sheet_name: str | int = 0,
    date_col: str = "date",
    surprise_col: str = "MP1_orthogonal",
) -> pd.DataFrame:
    """Load the Bauer-Swanson orthogonalized monetary-policy-surprise series.

    Returns columns [date: datetime64 (tz-naive, normalized), surprise: float],
    one row per meeting, dropping rows with missing surprise, sorted by date.
    Column names are parameters: confirm them against the actual SF Fed file
    (PRD §11) and pass overrides if they differ from the defaults.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]).dt.normalize(),
        "surprise": pd.to_numeric(raw[surprise_col], errors="coerce"),
    })
    return out.dropna(subset=["surprise"]).sort_values("date").reset_index(drop=True)
