# src/cbp/data/stance.py
from pathlib import Path
import pandas as pd

def stance_frame_from_scores(scores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Join per-statement stance scores [date, stance] onto the FOMC calendar to
    produce the release-aligned StanceSeries [release_date, release_ts, stance,
    doc_type]. Single source of truth for the calendar join (used by both the
    Phase 1 scored path and the Phase 0 CSV path).
    """
    s = scores.copy()
    s["release_date"] = pd.to_datetime(s["date"])
    merged = s.merge(calendar, on="release_date", how="inner")
    merged["doc_type"] = "statement"
    return merged[["release_date", "release_ts", "stance", "doc_type"]].sort_values(
        "release_ts"
    ).reset_index(drop=True)

def load_stance(path: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    scores = pd.DataFrame({"date": raw["date"], "stance": raw["stance"]})
    return stance_frame_from_scores(scores, calendar)
