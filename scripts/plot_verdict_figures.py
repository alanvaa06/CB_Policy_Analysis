"""Regenerate the Phase 2a verdict figure pack -> docs/results/figures/*.png.

Five figures behind docs/results/2026-06-29-lexicon-baseline-verdict.md:
  1 regime-confound.png    yearly avg rate vs lexicon tone (the confound)
  2 delta-r2-scorecard.png RoBERTa vs lexicon dR2, 6 cells
  3 era-jackknife.png      DGS1 h1 dR2 dropping each era (fragility)
  4 word-highlight.png     a sample statement with matched words colored
  5 lexicon-vs-roberta.png per-statement tone scatter (corr 0.41)

Figs 2-3 are constants from the verdict / scripts/lexicon_confound_check.py.
Figs 1,4,5 are recomputed from cached statements + FRED.

Usage: FRED_API_KEY=... python scripts/plot_verdict_figures.py   # needs .[infer is NOT required], .[viz]
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cbp.config import Config
from cbp.data.fred import FredClient
from cbp.data.fomc_statements import fetch_statements
from cbp.data.mp_surprise import load_surprise
from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon, tokenize, _count_side

HAWK_FILL, HAWK_TXT = "#FAECE7", "#712B13"
DOVE_FILL, DOVE_TXT = "#E6F1FB", "#0C447C"
CORAL, BLUE, GRAY = "#D85A30", "#378ADD", "#888780"
OUT = Path("docs/results/figures")


def _load():
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY.")
    market = FredClient(cfg.fred_api_key).fetch(["DGS1"], "1999-01-01", "2024-06-30")
    su = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                       sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    su = su[su["date"].dt.year >= 1999].reset_index(drop=True)
    statements = fetch_statements([d.date() for d in su["date"]], cfg.data_dir / "raw" / "statements")
    hawk, dove = load_lexicon(cfg.lexicon_path)
    lex = score_statements_lexicon(statements, hawk, dove)
    rob = pd.read_csv(cfg.data_dir / "raw" / "_phase1_stance_scores.csv", parse_dates=["date"])
    return cfg, market, statements, hawk, dove, lex, rob


def fig_regime(market, lex):
    lvl = market[["DGS1"]].dropna().sort_index()
    d = lex.copy().sort_values("date")
    d["utc"] = pd.to_datetime(d["date"]).dt.tz_localize("UTC")
    d = pd.merge_asof(d, lvl, left_on="utc", right_index=True, direction="backward")
    d["yr"] = d["date"].dt.year
    g = d.groupby("yr").agg(tone=("stance", "mean"), rate=("DGS1", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = [CORAL if t > 0.05 else (BLUE if t < -0.05 else GRAY) for t in g["tone"]]
    ax.scatter(g["rate"], g["tone"], c=cols, s=70, zorder=3)
    for r in g.itertuples():
        if r.tone > 0.4 or r.tone < -0.9 and r.rate > 0.5:
            ax.annotate(str(r.yr), (r.rate, r.tone), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("avg policy rate (DGS1, %)"); ax.set_ylabel("← dovish    lexicon tone    hawkish →")
    ax.set_title("Lexicon tone tracks the rate level (regime confound)")
    fig.tight_layout(); fig.savefig(OUT / "regime-confound.png", dpi=140); plt.close(fig)


def fig_scorecard():
    cells = ["DGS2·1d", "DGS2·5d", "DGS2·22d", "DGS1·1d", "DGS1·5d", "DGS1·22d"]
    rob = [-0.0029, -0.0066, -0.0162, -0.0056, -0.0124, -0.0023]
    lex = [0.0116, -0.0152, -0.0363, 0.0552, -0.0004, -0.0520]
    import numpy as np
    x = np.arange(len(cells)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, rob, w, label="RoBERTa (AI)", color=GRAY)
    ax.bar(x + w / 2, lex, w, label="lexicon (word-count)", color=CORAL)
    ax.axhline(0, color="#c3c2b7", lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(cells)
    ax.set_ylabel("ΔR²  (with tone − without)"); ax.legend()
    ax.set_title("Does tone add OOS value beyond the surprise?  (ΔR²>0 = yes)")
    fig.tight_layout(); fig.savefig(OUT / "delta-r2-scorecard.png", dpi=140); plt.close(fig)


def fig_jackknife():
    labs = ["full\nsample", "drop\npre-2008", "drop\nZLB 08–15", "drop\n2016–19", "drop\n2020–23"]
    v = [0.0552, -0.0407, 0.0610, 0.0536, 0.0407]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labs, v, color=[CORAL if x >= 0 else BLUE for x in v])
    ax.axhline(0, color="#c3c2b7", lw=1.5)
    ax.set_ylabel("DGS1·1d ΔR²")
    ax.set_title("The h=1 signal flips negative without pre-2008 (fragile)")
    fig.tight_layout(); fig.savefig(OUT / "era-jackknife.png", dpi=140); plt.close(fig)


def fig_words(statements, hawk, dove):
    statements = statements.copy()
    statements["hits"] = statements["text"].apply(
        lambda t: _count_side(tokenize(t), hawk) + _count_side(tokenize(t), dove))
    best = statements.sort_values("hits", ascending=False).iloc[0]
    text = " ".join(best["text"].split())[:1100]
    words = text.split()
    ncols = 92
    lines, cur, cl = [], [], 0
    for wd in words:
        add = len(wd) + (1 if cur else 0)
        if cl + add > ncols:
            lines.append(cur); cur, cl = [], 0; add = len(wd)
        cur.append(wd); cl += add
    if cur:
        lines.append(cur)
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(lines) + 1))
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    cw, lh = 1.0 / ncols, 1.0 / (len(lines) + 1)
    for li, line in enumerate(lines):
        col = 0
        y = 1 - (li + 0.7) * lh
        for wd in line:
            w = "".join(c for c in wd.lower() if c.isalpha())
            kw = dict(family="monospace", fontsize=9, va="center", ha="left")
            if w and any(w.startswith(s) for s in hawk):
                ax.text(col * cw, y, wd, color=HAWK_TXT, bbox=dict(boxstyle="round,pad=0.1", fc=HAWK_FILL, ec="none"), **kw)
            elif w and any(w.startswith(s) for s in dove):
                ax.text(col * cw, y, wd, color=DOVE_TXT, bbox=dict(boxstyle="round,pad=0.1", fc=DOVE_FILL, ec="none"), **kw)
            else:
                ax.text(col * cw, y, wd, color="#2c2c2a", **kw)
            col += len(wd) + 1
    ax.set_title(f"FOMC statement {pd.to_datetime(best['date']).date()} — matched words describe the economy, not policy action",
                 fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "word-highlight.png", dpi=140); plt.close(fig)


def fig_lex_vs_rob(lex, rob):
    sc = lex.merge(rob.rename(columns={"stance": "rob"}), on="date", how="inner")
    corr = float(sc["stance"].corr(sc["rob"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sc["rob"], sc["stance"], s=22, color=BLUE, alpha=0.45)
    ax.axhline(0, color="#e1e0d9", lw=1); ax.axvline(0, color="#e1e0d9", lw=1)
    ax.set_xlabel("RoBERTa tone (graded)"); ax.set_ylabel("lexicon tone (3 bands)")
    ax.set_title(f"Two readers, weak agreement (corr = {corr:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "lexicon-vs-roberta.png", dpi=140); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, market, statements, hawk, dove, lex, rob = _load()
    fig_regime(market, lex)
    fig_scorecard()
    fig_jackknife()
    fig_words(statements, hawk, dove)
    fig_lex_vs_rob(lex, rob)
    print("wrote 5 figures to", OUT)


if __name__ == "__main__":
    main()
