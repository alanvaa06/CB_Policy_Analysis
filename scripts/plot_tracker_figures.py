"""Static figures for the README research write-up, from data/monitor/tone_history.csv.

Descriptive only (see PRD 005 / the Statement Tracker). Produces:
  docs/results/figures/tracker-theme-heatmap.png   themes x time intensity (per 1k words)
  docs/results/figures/tracker-change-magnitude.png how much each statement was rewritten
  docs/results/figures/tracker-comm-style.png       length / readability / uncertainty over time

Usage: python scripts/plot_tracker_figures.py   (needs data/monitor/tone_history.csv; .[viz])
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("docs/results/figures")
THEMES = [
    ("theme_inflation", "inflation"),
    ("theme_employment", "employment"),
    ("theme_growth", "growth"),
    ("theme_balance_sheet", "balance sheet"),
    ("theme_financial_conditions", "financial conditions"),
]


def _load() -> pd.DataFrame:
    d = pd.read_csv("data/monitor/tone_history.csv", parse_dates=["date"])
    return d.sort_values("date").reset_index(drop=True)


def theme_heatmap(d: pd.DataFrame) -> None:
    z = np.array([d[c].to_numpy() for c, _ in THEMES])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = mdates.date2num(d["date"])
    im = ax.imshow(z, aspect="auto", cmap="YlOrRd", origin="upper",
                   extent=[x.min(), x.max(), len(THEMES) - 0.5, -0.5])
    ax.set_yticks(range(len(THEMES)))
    ax.set_yticklabels([name for _, name in THEMES])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("What the FOMC talks about — theme intensity (mentions per 1,000 words), 1999–2026")
    fig.colorbar(im, ax=ax, label="per 1k words", pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG / "tracker-theme-heatmap.png", dpi=140)
    plt.close(fig)


def change_magnitude(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(d["date"], d["change_magnitude"], color="#7A3B8F", lw=1.0, marker="o", ms=2.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("edit fraction (0–1)")
    ax.set_title("How much each FOMC statement was rewritten vs the prior meeting")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "tracker-change-magnitude.png", dpi=140)
    plt.close(fig)


def comm_style(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True)
    series = [
        ("word_count", "length (words)", "#2C6E9B"),
        ("flesch", "readability (Flesch)", "#B4661E"),
        ("uncertainty_per1k", "uncertainty (per 1k words)", "#3F8F5B"),
    ]
    for ax, (col, label, color) in zip(axes, series):
        ax.plot(d["date"], d[col], color=color, lw=1.1)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_title("FOMC communication style over time — length, readability, uncertainty")
    fig.tight_layout()
    fig.savefig(FIG / "tracker-comm-style.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = _load()
    theme_heatmap(d)
    change_magnitude(d)
    comm_style(d)
    print(f"wrote tracker figures to {FIG} ({len(d)} statements)")


if __name__ == "__main__":
    main()
