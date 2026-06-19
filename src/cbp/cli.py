# src/cbp/cli.py
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.align.aligner import build_aligned_panel
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward
from cbp.eval.metrics import oos_r2, rmse, hit_rate, sign_test
from cbp.eval.eventstudy import event_study

def run_report(market: pd.DataFrame, stance: pd.DataFrame, config: Config) -> dict:
    panel = build_aligned_panel(market, stance, config)
    oos = {}
    for sid in config.target_series:
        for h in config.horizons:
            col = f"{sid}_h{h}"
            if col not in panel.columns or len(panel) <= config.n0:
                continue
            wf = run_walkforward(panel, col, SimpleOLS(), ZeroChange(), config.n0)
            if wf.empty:
                continue
            yt, yp, yb = wf["y_true"].to_numpy(), wf["y_pred"].to_numpy(), wf["y_base"].to_numpy()
            oos[(sid, h)] = {
                "n": len(wf), "oos_r2": oos_r2(yt, yp, yb), "rmse": rmse(yt, yp),
                "hit_rate": hit_rate(yt, yp), **{f"sign_{k}": v for k, v in sign_test(yt, yp).items()},
            }
    events = {sid: event_study(market, stance, sid, config.event_window)
              for sid in config.target_series if sid in market.columns}
    return {"oos": oos, "events": events}

def _print_report(report: dict) -> None:
    print("\n=== OOS walk-forward ===")
    for (sid, h), m in report["oos"].items():
        print(f"{sid} h={h:>2}: n={m['n']:>3}  OOS_R2={m['oos_r2']:+.3f}  "
              f"hit={m['hit_rate']:.2f}  sign_p={m['sign_pvalue']:.3f}")
    print("\n=== Event study [t-1,t+1] ===")
    for sid, e in report["events"].items():
        print(f"{sid}: slope={e['slope']:+.4f}  t={e['tstat']:+.2f}  r2={e['r2']:.3f}  n={e['n']}")

def main() -> None:
    ap = argparse.ArgumentParser(description="FOMC stance eval harness (Phase 0)")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2022-12-31")
    args = ap.parse_args()
    from cbp.data.fred import FredClient
    from cbp.data.fomc_calendar import load_fomc_calendar
    from cbp.data.stance import load_stance
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY to run the live report.")
    market = FredClient(cfg.fred_api_key).fetch(list(cfg.target_series), args.start, args.end)
    cal = load_fomc_calendar(cfg.data_dir / "raw" / "fomc_dates.csv")
    stance = load_stance(cfg.data_dir / "raw" / "tdw_stance.csv", cal)
    _print_report(run_report(market, stance, cfg))

if __name__ == "__main__":
    main()
