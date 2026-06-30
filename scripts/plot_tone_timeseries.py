"""Plot lexicon vs FOMC-RoBERTa statement tone over time + correlation scatter.

Usage: python scripts/plot_tone_timeseries.py
Reads the cached RoBERTa scores and recomputes lexicon scores from cached
statements, writes docs/results/figures/tone-timeseries.png.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from cbp.viz.tone_compare import build_tone_comparison


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cbp.config import Config
    from cbp.data.fomc_statements import fetch_statements
    from cbp.data.mp_surprise import load_surprise
    from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon

    cfg = Config()
    surprise = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                             sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    surprise = surprise[surprise["date"].dt.year >= 1999].reset_index(drop=True)
    statements = fetch_statements([d.date() for d in surprise["date"]], cfg.data_dir / "raw" / "statements")
    hawk, dove = load_lexicon(cfg.lexicon_path)
    lex = score_statements_lexicon(statements, hawk, dove)

    rob = pd.read_csv(cfg.data_dir / "raw" / "_phase1_stance_scores.csv", parse_dates=["date"])
    merged, corr = build_tone_comparison(lex, rob[["date", "stance"]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(merged["date"], merged["stance_roberta"], label="FOMC-RoBERTa", lw=1.2)
    ax1.plot(merged["date"], merged["stance_lexicon"], label="lexicon", lw=1.2)
    ax1.axhline(0, color="0.6", lw=0.8); ax1.legend(); ax1.set_title("FOMC statement tone over time")
    ax2.scatter(merged["stance_roberta"], merged["stance_lexicon"], s=12)
    ax2.set_xlabel("RoBERTa"); ax2.set_ylabel("lexicon"); ax2.set_title(f"corr = {corr:.2f}")
    out = Path("docs/results/figures/tone-timeseries.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}  (n={len(merged)}, corr={corr:.3f})")


if __name__ == "__main__":
    main()
