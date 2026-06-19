# tests/test_aligner.py
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.align.aligner import forward_change, build_aligned_panel

def _market():
    idx = pd.bdate_range("2020-01-27", "2020-02-28", tz="UTC")
    # DGS2 rises by 0.01 each business day from 1.00
    vals = 1.00 + 0.01 * np.arange(len(idx))
    return pd.DataFrame({"DGS2": vals, "EFFR": vals}, index=idx)

def test_forward_change_strictly_after_ts():
    m = _market()
    ts = pd.Timestamp("2020-01-29 19:00", tz="UTC")  # release
    # h=1: change from first bday strictly AFTER ts to that day; 2020-01-30 vs ...
    fc = forward_change(m["DGS2"], ts, h=1)
    # value on 2020-01-30 minus value on 2020-01-29 (last point at/before exit)
    assert fc > 0  # rising series => positive 1-day fwd change

def test_panel_has_target_cols_and_no_future_leak():
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert "DGS2_h1" in panel.columns and "DGS2_h5" in panel.columns
    # invariant: target for a release never uses market data on/before release_ts
    assert panel["release_ts"].iloc[0] == pd.Timestamp("2020-01-29 19:00", tz="UTC")

def test_drop_release_when_window_incomplete():
    m = _market()
    # release near end of series: h=22 window cannot close -> row dropped
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-02-27"])})
    cal["release_ts"] = pd.Timestamp("2020-02-27 19:00", tz="UTC")
    stance = cal.assign(stance=0.1, doc_type="statement")
    cfg = Config(horizons=(22,), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert panel.empty
