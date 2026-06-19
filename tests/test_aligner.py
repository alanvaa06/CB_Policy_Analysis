# tests/test_aligner.py
import logging

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
    # base = 2020-01-29 close (1.02); h=1 future = 2020-01-30 close (1.03).
    # Pin the EXACT value, not just direction: an off-by-one that drops the
    # future leg onto the wrong bar (or ignores ts) shifts this away from 0.01.
    fc = forward_change(m["DGS2"], ts, h=1)
    assert abs(fc - 0.01) < 1e-9
    # h=5 future = 2020-02-05 close (1.07): base 1.02 -> 0.05.
    assert abs(forward_change(m["DGS2"], ts, h=5) - 0.05) < 1e-9

def test_forward_change_excludes_release_day_bar():
    # Boundary case: a market bar sits EXACTLY at release_ts, and steps are
    # non-uniform so bar identity is observable. The leak-safety invariant is
    # base = bar AT ts, future = first bar STRICTLY AFTER ts. An off-by-one that
    # treats the release-day bar as the future leg (index >= ts) would read 1.02
    # as future and the prior bar as base -> 0.01 instead of the correct 0.08.
    idx = pd.DatetimeIndex(
        [
            "2020-01-28 19:00",
            "2020-01-29 19:00",  # == release_ts
            "2020-01-30 19:00",
            "2020-01-31 19:00",
            "2020-02-03 19:00",
            "2020-02-04 19:00",
        ],
        tz="UTC",
    )
    vals = np.array([1.01, 1.02, 1.10, 1.11, 1.12, 1.13])
    s = pd.Series(vals, index=idx, name="DGS2")
    ts = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    # base = 1.02 (the at-ts bar), future h=1 = 1.10 (first bar strictly after ts).
    assert abs(forward_change(s, ts, h=1) - 0.08) < 1e-9
    # h=2 future = 1.11 -> 0.09.
    assert abs(forward_change(s, ts, h=2) - 0.09) < 1e-9

def test_panel_has_target_cols_and_no_future_leak():
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert "DGS2_h1" in panel.columns and "DGS2_h5" in panel.columns
    assert panel["release_ts"].iloc[0] == pd.Timestamp("2020-01-29 19:00", tz="UTC")
    # invariant: target for a release never uses market data on/before release_ts.
    # Pin the EXACT target so an off-by-one window can't pass on structure alone.
    assert abs(panel["DGS2_h1"].iloc[0] - 0.01) < 1e-9
    assert abs(panel["DGS2_h5"].iloc[0] - 0.05) < 1e-9

def test_panel_target_invariant_to_future_perturbation():
    # Mirrors the Task-10 perturbation invariant at the alignment layer: the
    # h=1 target reads exactly one future bar (2020-01-30) against the base bar
    # (2020-01-29). Corrupting a STRICTLY-FUTURE bar must move the target;
    # corrupting a STRICTLY-PAST bar (before the base) must leave it bit-identical.
    m = _market()
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1,), target_series=("DGS2",))
    base = build_aligned_panel(m, stance, cfg)["DGS2_h1"].iloc[0]

    m_future = m.copy()
    m_future.loc[pd.Timestamp("2020-01-30", tz="UTC"), "DGS2"] = 5.0
    future = build_aligned_panel(m_future, stance, cfg)["DGS2_h1"].iloc[0]
    assert abs(future - base) > 1e-9  # future bar feeds the target -> changes

    m_past = m.copy()
    m_past.loc[pd.Timestamp("2020-01-28", tz="UTC"), "DGS2"] = -5.0
    past = build_aligned_panel(m_past, stance, cfg)["DGS2_h1"].iloc[0]
    assert past == base  # pre-base bar is never read -> exactly unchanged

def test_panel_handles_tz_naive_market_index():
    # Real FRED data has a tz-NAIVE DatetimeIndex, but release_ts is tz-aware UTC.
    # The live run crashed here (pandas InvalidComparison). build_aligned_panel
    # must normalize a tz-naive market index to UTC and still produce correct
    # targets identical to the tz-aware case.
    idx = pd.bdate_range("2020-01-27", "2020-02-28")  # tz-naive, like real FRED
    vals = 1.00 + 0.01 * np.arange(len(idx))
    market = pd.DataFrame({"DGS2": vals}, index=idx)
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-01-29"])})
    cal["release_ts"] = pd.Timestamp("2020-01-29 19:00", tz="UTC")
    stance = cal.assign(stance=0.5, doc_type="statement")
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    panel = build_aligned_panel(market, stance, cfg)  # must not raise
    assert len(panel) == 1
    assert abs(panel["DGS2_h1"].iloc[0] - 0.01) < 1e-9
    assert abs(panel["DGS2_h5"].iloc[0] - 0.05) < 1e-9

def test_drop_release_when_window_incomplete():
    m = _market()
    # release near end of series: h=22 window cannot close -> row dropped
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-02-27"])})
    cal["release_ts"] = pd.Timestamp("2020-02-27 19:00", tz="UTC")
    stance = cal.assign(stance=0.1, doc_type="statement")
    cfg = Config(horizons=(22,), target_series=("DGS2",))
    panel = build_aligned_panel(m, stance, cfg)
    assert panel.empty

def test_dropped_release_logs_reason(caplog):
    # PRD §7: a release whose target window cannot close must be dropped WITH a
    # logged reason naming the release_ts and the (series, h) window that failed.
    m = _market()
    release_ts = pd.Timestamp("2020-02-27 19:00", tz="UTC")
    cal = pd.DataFrame({"release_date": pd.to_datetime(["2020-02-27"])})
    cal["release_ts"] = release_ts
    stance = cal.assign(stance=0.1, doc_type="statement")
    cfg = Config(horizons=(22,), target_series=("DGS2",))
    with caplog.at_level(logging.WARNING, logger="cbp.align.aligner"):
        panel = build_aligned_panel(m, stance, cfg)
    assert panel.empty  # behavior unchanged: still dropped
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING log for the dropped release"
    msg = " ".join(r.getMessage() for r in warnings)
    assert str(release_ts) in msg          # names which release was dropped
    assert "DGS2" in msg and "22" in msg   # names the (series, h) target window
