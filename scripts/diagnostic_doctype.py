"""Per-document-type diagnostic for the Phase 0 eval harness.

Runs the SAME harness (run_report) on three throwaway TDW stance series —
combined (466 dates, minutes+pressers+speeches), mm-only (meeting minutes,
214), pc-only (press conferences, 63) — against FRED DGS2/EFFR, and prints the
OOS metrics side by side. Read-only research diagnostic; does not modify the
committed harness. Requires FRED_API_KEY in the environment.

Run: set -a && . ./.env && set +a && python scripts/diagnostic_doctype.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from cbp.cli import run_report
from cbp.config import Config
from cbp.data.fomc_calendar import load_fomc_calendar
from cbp.data.fred import FredClient
from cbp.data.stance import load_stance

RAW = Path("data/raw")
DOCT = RAW / "doctype"


def build_doctype_csvs() -> dict[str, Path]:
    """Build date,stance CSVs for mm-only and pc-only from the TDW aggregate files."""
    mm = pd.read_excel(DOCT / "aggregate_measure_mm.xlsx")
    pc = pd.read_excel(DOCT / "aggregate_measure_pc.xlsx")
    mm_df = pd.DataFrame(
        {
            "date": pd.to_datetime(mm["ReleaseDate"]).dt.strftime("%Y-%m-%d"),
            "stance": mm["our_measure"].astype(float),
        }
    )
    pc_df = pd.DataFrame(
        {
            "date": pd.to_datetime(pc["EndDate"], format="%B/%d/%Y").dt.strftime("%Y-%m-%d"),
            "stance": pc["our_measure"].astype(float),
        }
    )
    out: dict[str, Path] = {}
    for name, df in (("mm", mm_df), ("pc", pc_df)):
        clean = df.dropna().drop_duplicates("date").sort_values("date")
        p = DOCT / f"stance_{name}.csv"
        clean.to_csv(p, index=False)
        out[name] = p
    return out


def run_variant(stance_csv: Path, cfg: Config, market: pd.DataFrame) -> dict:
    s = pd.read_csv(stance_csv)
    dates_csv = stance_csv.with_name(stance_csv.stem + "_dates.csv")
    pd.DataFrame({"release_date": s["date"]}).drop_duplicates().to_csv(dates_csv, index=False)
    cal = load_fomc_calendar(dates_csv)
    stance = load_stance(stance_csv, cal)
    return run_report(market, stance, cfg)


def print_report(name: str, report: dict) -> None:
    print(f"\n===== {name} =====")
    for (sid, h), m in report["oos"].items():
        print(
            f"  {sid} h={h:>2}: n={m['n']:>3}  OOS_R2={m['oos_r2']:+.3f}  "
            f"hit={m['hit_rate']:.2f}  sign_p={m['sign_pvalue']:.3f}"
        )
    for sid, e in report["events"].items():
        print(f"  event {sid}: slope={e['slope']:+.4f}  t={e['tstat']:+.2f}  r2={e['r2']:.3f}  n={e['n']}")


def main() -> None:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("FRED_API_KEY not set (run: set -a && . ./.env && set +a).")
    cfg = Config(fred_api_key=key)
    market = FredClient(key).fetch(list(cfg.target_series), "1996-01-01", "2022-12-31")

    doctype = build_doctype_csvs()
    variants = {
        "combined (466: mm+pc+sp)": RAW / "tdw_stance.csv",
        "mm-only (minutes, 214)": doctype["mm"],
        "pc-only (pressers, 63)": doctype["pc"],
    }
    for name, csv in variants.items():
        print_report(name, run_variant(csv, cfg, market))


if __name__ == "__main__":
    main()
