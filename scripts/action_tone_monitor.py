"""Descriptive hike/cut/hold tracker from FOMC statement action verbs.

Scores every FOMC statement 1999-present with the MONITORING-ONLY action lexicon
(data/lexicons/action_tone.json: +1 'raise', -1 'lower', 0 hold) and writes
docs/results/figures/action-tone-tracker.png + prints the most recent meetings.

This index mirrors the policy DECISION itself — it is descriptive, NOT predictive
(redundant with any rate surprise by construction). See the Phase 2a verdict.

Usage: python scripts/action_tone_monitor.py   (needs cached statements + BS xlsx; .[viz])
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cbp.config import Config
from cbp.data.fomc_statements import fetch_statements
from cbp.data.mp_surprise import load_surprise
from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon

# FOMC announcement dates not covered by the Bauer-Swanson file (2024-2026, from the Fed calendar)
RECENT_DATES = [
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
    dates = [d.date() for d in su["date"]] + [pd.Timestamp(s).date() for s in RECENT_DATES]
    statements = fetch_statements(dates, cfg.data_dir / "raw" / "statements")

    hawk, dove = load_lexicon(cfg.lexicon_path.parent / "action_tone.json")
    sc = score_statements_lexicon(statements, hawk, dove).sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    col = sc["stance"].map(lambda v: "#D85A30" if v > 0 else ("#378ADD" if v < 0 else "#B4B2A9"))
    ax.bar(sc["date"], sc["stance"], width=40, color=col)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_ylim(-1.2, 1.2); ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["cut", "hold", "hike"])
    ax.set_title("FOMC action tracker (statement verb): hike / hold / cut, 1999-2026")
    out = Path("docs/results/figures/action-tone-tracker.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)

    rec = sc[sc["date"] >= "2024-01-01"]
    lab = {1.0: "HIKE", -1.0: "CUT", 0.0: "hold"}
    print(f"wrote {out}\n\nrecent meetings (action tracker):")
    for r in rec.itertuples():
        print(f"  {pd.to_datetime(r.date).date()}  {lab.get(r.stance, f'{r.stance:+.2f}')}")


if __name__ == "__main__":
    main()
