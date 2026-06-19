# tests/test_eventstudy.py
import numpy as np
import pandas as pd
import pytest
from cbp.eval.eventstudy import event_study


def _constant_slope_fixture(k: float):
    """Build a market series whose [t-1, t+1] window change equals exactly k*stance
    for each release, so event_study must recover slope ~= k.

    For window (-1, 1), _window_change = series[first bday after t] - series[t-1 bday].
    Every series point is anchored at 0.0; for each release the post-event point
    (first bday strictly after t) is set to k*stance, leaving the pre-event anchor at 0.
    """
    idx = pd.bdate_range("2020-01-01", "2020-12-31", tz="UTC")
    s = pd.Series(0.0, index=idx, name="DGS2")
    releases = pd.to_datetime(
        ["2020-03-18", "2020-06-10", "2020-09-16"]
    ).tz_localize("UTC")
    stances = [0.5, -0.2, 0.1]
    for ts, st in zip(releases, stances):
        post = s.index[s.index > ts][0]  # first bday strictly after t
        s.loc[post] = k * st
    rel = pd.DataFrame({"release_ts": releases, "stance": stances})
    return s.to_frame(), rel


def test_event_study_recovers_known_slope():
    k = 1.5
    market, rel = _constant_slope_fixture(k)
    out = event_study(market, rel, "DGS2", window=(-1, 1))
    assert {"slope", "tstat", "r2", "n"} <= set(out)
    assert out["n"] == 3
    # change == k*stance exactly by construction -> perfect linear fit
    assert out["slope"] == pytest.approx(k, rel=1e-9, abs=1e-9)
    assert out["r2"] == pytest.approx(1.0, rel=1e-9, abs=1e-9)


def test_event_study_constant_stance_returns_nan_not_raises():
    # All surviving stance values identical -> singular (constant) regressor.
    # Must degrade to NaN, not raise IndexError.
    market, rel = _constant_slope_fixture(1.5)
    rel = rel.assign(stance=[0.3, 0.3, 0.3])
    out = event_study(market, rel, "DGS2", window=(-1, 1))
    assert out["n"] == 3
    assert np.isnan(out["slope"])
    assert np.isnan(out["tstat"])
    assert np.isnan(out["r2"])
