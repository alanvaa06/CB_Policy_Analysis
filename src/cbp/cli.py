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
from cbp.eval.nested import nested_oos, residual_stance_regression

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

def run_nested_report(market: pd.DataFrame, stance: pd.DataFrame, surprise: pd.DataFrame, config: Config) -> dict:
    panel = build_aligned_panel(market, stance, config, extra_features=surprise)
    nested, residual = {}, {}
    for sid in config.target_series:
        for h in config.horizons:
            col = f"{sid}_h{h}"
            if col not in panel.columns or len(panel) <= config.n0:
                continue
            nested[(sid, h)] = nested_oos(panel, col, config.n0)
            residual[(sid, h)] = residual_stance_regression(panel, col, config.n0)
    return {"nested": nested, "residual": residual, "n_releases": len(panel)}

def _print_nested(report: dict) -> None:
    print(f"\n=== Nested OOS (surprise-only vs surprise+stance) | aligned releases: {report['n_releases']} ===")
    print(f"{'target':>8} {'h':>3} {'n':>4} {'R2_base':>9} {'R2_full':>9} {'dR2':>9} {'stance_t':>9}")
    for (sid, h), m in report["nested"].items():
        print(f"{sid:>8} {h:>3} {m['n']:>4} {m['r2_base']:>+9.4f} {m['r2_full']:>+9.4f} "
              f"{m['delta_r2']:>+9.4f} {m['stance_partial_t']:>+9.2f}")
    print("\n=== Residual event-study (residual ~ stance, after surprise) ===")
    for (sid, h), e in report["residual"].items():
        print(f"{sid:>8} h={h:>2}: slope={e['slope']:+.4f}  t={e['tstat']:+.2f}  r2={e['r2']:.3f}  n={e['n']}")

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FOMC stance eval harness")
    ap.add_argument("--mode", choices=["phase0", "phase1"], default="phase1")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--model", default=None,
                    help="override RoBERTa model id (default: config.roberta_model_id)")
    ap.add_argument("--tone-method", choices=["roberta", "lexicon"], default="roberta",
                    dest="tone_method",
                    help="stance source for --mode phase1 (default: roberta)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    from cbp.data.fred import FredClient
    cfg = Config(fred_api_key=os.environ.get("FRED_API_KEY"))
    if not cfg.fred_api_key:
        raise SystemExit("Set FRED_API_KEY to run the live report.")
    market = FredClient(cfg.fred_api_key).fetch(list(cfg.target_series), args.start, args.end)

    if args.mode == "phase0":
        from cbp.data.fomc_calendar import load_fomc_calendar
        from cbp.data.stance import load_stance
        cal = load_fomc_calendar(cfg.data_dir / "raw" / "fomc_dates.csv")
        stance = load_stance(cfg.data_dir / "raw" / "tdw_stance.csv", cal)
        _print_report(run_report(market, stance, cfg))
        return

    # phase1: the canonical FOMC announcement dates AND the orthogonalized surprise
    # come from one source (the Bauer-Swanson file) so statement fetching, the
    # release calendar, and the control align 1:1. The TDW fomc_dates.csv is a
    # multi-doc-type calendar (statements+minutes+speeches) and must NOT drive
    # fetching — doing so scrapes misdated non-statements and misses real meetings.
    from cbp.data.fomc_statements import fetch_statements
    from cbp.models.stance_scorer import score_statements, load_fomc_roberta
    from cbp.data.stance import stance_frame_from_scores
    from cbp.data.mp_surprise import load_surprise

    surprise = load_surprise(
        cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
        sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH",
    )
    surprise = surprise[surprise["date"].dt.year >= 1999].reset_index(drop=True)
    # Release calendar from the same dates: announcement ~14:00 ET -> UTC (mirrors
    # load_fomc_calendar so the leak-safe forward windows are unchanged).
    ts_et = (surprise["date"] + pd.Timedelta(hours=14)).dt.tz_localize("America/New_York")
    cal = pd.DataFrame({"release_date": surprise["date"], "release_ts": ts_et.dt.tz_convert("UTC")})

    statements = fetch_statements([d.date() for d in surprise["date"]], cfg.data_dir / "raw" / "statements")
    if args.tone_method == "lexicon":
        from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
        hawk, dove = load_lexicon(cfg.lexicon_path)
        scores = score_statements_lexicon(statements, hawk, dove)
    else:
        scores = score_statements(statements, load_fomc_roberta(args.model or cfg.roberta_model_id))
    stance = stance_frame_from_scores(scores, cal)
    _print_nested(run_nested_report(market, stance, surprise, cfg))

if __name__ == "__main__":
    main()
