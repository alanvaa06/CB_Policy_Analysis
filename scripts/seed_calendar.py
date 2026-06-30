# scripts/seed_calendar.py
"""One-time generator for data/monitor/fomc_calendar.csv.

Unions the Bauer-Swanson announcement dates (1999-2023, from the surprise xlsx) with
the post-2023 meeting dates the BS file doesn't cover, and writes a single sorted
`date` column. This file REPLACES the hand-maintained RECENT_DATES list in
scripts/action_tone_monitor.py — extend POST_2023 once a year from the Fed calendar.

Usage: python scripts/seed_calendar.py   (needs the BS xlsx under data/raw/; .[dev])
"""
from pathlib import Path

import pandas as pd

from cbp.config import Config
from cbp.data.mp_surprise import load_surprise

POST_2023 = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
    "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18",
    "2026-04-29", "2026-06-17",
]


def main() -> None:
    cfg = Config()
    su = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                       sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    su = su[su["date"].dt.year >= 1999]
    dates = sorted({d.date() for d in su["date"]} | {pd.Timestamp(s).date() for s in POST_2023})
    out = Path("data/monitor/fomc_calendar.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": [d.isoformat() for d in dates]}).to_csv(out, index=False)
    print(f"wrote {out} with {len(dates)} meeting dates ({dates[0]} … {dates[-1]})")


if __name__ == "__main__":
    main()
