"""Adversarial hardening of the Phase 2a lexicon h=1 OOS result: is it a
policy-regime confound? (Referenced by docs/results/2026-06-29-lexicon-baseline-verdict.md.)

For the two positive cells (DGS1_h1, DGS2_h1) compares stance's marginal OOS R2:
  A=[surprise]  B=[surprise,stance]  C=[surprise,regime]  D=[surprise,regime,stance]
  regime = front-end rate LEVEL at release (cycle position; leak-free, known at release).
If (B-A)>0 but (D-C)<=0  -> stance is subsumed by regime -> CONFOUND.
Also prints stance-by-year (regime proxy) and an era jackknife of DGS1_h1 (B-A).

Usage: FRED_API_KEY=... python scripts/lexicon_confound_check.py
"""
import os
import pandas as pd
from cbp.config import Config
from cbp.data.fred import FredClient
from cbp.data.fomc_statements import fetch_statements
from cbp.data.mp_surprise import load_surprise
from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
from cbp.data.stance import stance_frame_from_scores
from cbp.align.aligner import build_aligned_panel
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward
from cbp.eval.metrics import oos_r2


def main() -> None:
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY.")
    market = FredClient(cfg.fred_api_key).fetch(list(cfg.target_series), "1999-01-01", "2024-06-30")

    surprise = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                             sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    surprise = surprise[surprise["date"].dt.year >= 1999].reset_index(drop=True)
    ts_et = (surprise["date"] + pd.Timedelta(hours=14)).dt.tz_localize("America/New_York")
    cal = pd.DataFrame({"release_date": surprise["date"], "release_ts": ts_et.dt.tz_convert("UTC")})

    statements = fetch_statements([d.date() for d in surprise["date"]], cfg.data_dir / "raw" / "statements")
    hawk, dove = load_lexicon(cfg.lexicon_path)
    stance = stance_frame_from_scores(score_statements_lexicon(statements, hawk, dove), cal)
    panel = build_aligned_panel(market, stance, cfg, extra_features=surprise)

    lvl = market[["DGS1"]].rename(columns={"DGS1": "regime"}).dropna().sort_index()
    panel = panel.sort_values("release_ts").reset_index(drop=True)
    panel = pd.merge_asof(panel, lvl, left_on="release_ts", right_index=True, direction="backward")

    def r2_for(target, feats):
        wf = run_walkforward(panel, target, SimpleOLS(), ZeroChange(), cfg.n0, feature_cols=feats)
        return oos_r2(wf["y_true"].to_numpy(), wf["y_pred"].to_numpy(), wf["y_base"].to_numpy())

    print("stance dist:", panel["stance"].round(2).value_counts().to_dict())
    print(f"corr(stance, regime-level) = {panel['stance'].corr(panel['regime']):+.3f}\n")
    print(f"{'cell':>9} {'A surp':>9} {'B +st':>9} {'C +reg':>9} {'D +rg+st':>9} {'B-A':>8} {'D-C':>8} {'C-A':>8}")
    for t in ["DGS1_h1", "DGS2_h1", "DGS1_h22", "DGS2_h22"]:
        a, b = r2_for(t, ["surprise"]), r2_for(t, ["surprise", "stance"])
        c, d = r2_for(t, ["surprise", "regime"]), r2_for(t, ["surprise", "regime", "stance"])
        print(f"{t:>9} {a:>+9.4f} {b:>+9.4f} {c:>+9.4f} {d:>+9.4f} {b-a:>+8.4f} {d-c:>+8.4f} {c-a:>+8.4f}")

    panel["yr"] = panel["release_ts"].dt.year
    print("\nstance mean by year:")
    print(panel.groupby("yr")["stance"].agg(["mean", "count"]).round(2).to_string())

    print("\nera jackknife of DGS1_h1 (B-A), dropping each era:")
    for name, (lo, hi) in {"pre2008": (1999, 2007), "ZLB 08-15": (2008, 2015),
                           "16-19": (2016, 2019), "20-23": (2020, 2023)}.items():
        sub = panel[(panel["yr"] < lo) | (panel["yr"] > hi)].reset_index(drop=True)

        def r2s(feats):
            wf = run_walkforward(sub, "DGS1_h1", SimpleOLS(), ZeroChange(), cfg.n0, feature_cols=feats)
            return oos_r2(wf["y_true"].to_numpy(), wf["y_pred"].to_numpy(), wf["y_base"].to_numpy())

        print(f"  drop {name:>10}: B-A = {r2s(['surprise','stance']) - r2s(['surprise']):+.4f}")


if __name__ == "__main__":
    main()
