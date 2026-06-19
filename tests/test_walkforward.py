# tests/test_walkforward.py
import numpy as np
import pandas as pd
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward

def _panel(n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    stance = rng.normal(size=n)
    target = (2.0 * stance if signal else rng.normal(size=n)) + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"release_ts": ts, "stance": stance, "DGS2_h1": target})

def test_skips_until_n0_then_predicts():
    p = _panel(30)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    assert len(out) == 10                      # 30 - 20 OOS predictions
    assert list(out.columns) == ["release_ts", "y_true", "y_pred", "y_base"]

def test_recovers_known_signal():
    p = _panel(120, signal=True)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert corr > 0.9                          # strong OOS skill on real signal

def test_rejects_null_signal():
    p = _panel(120, signal=False)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert abs(corr) < 0.4                      # no spurious skill on noise

def test_no_lookahead_future_perturbation_invariant():
    p = _panel(60, signal=True, seed=1)
    out1 = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    # corrupt the LAST row's target only; predictions for earlier rows must not change
    p2 = p.copy(); p2.loc[p2.index[-1], "DGS2_h1"] = 999.0
    out2 = run_walkforward(p2, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    pd.testing.assert_series_equal(
        out1["y_pred"].iloc[:-1].reset_index(drop=True),
        out2["y_pred"].iloc[:-1].reset_index(drop=True),
    )
