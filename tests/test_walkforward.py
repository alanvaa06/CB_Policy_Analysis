# tests/test_walkforward.py
import logging

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

def test_n0_skip_logs_count(caplog):
    # PRD §7: leading releases below the N0 minimum are skipped; the count must
    # be logged (INFO) so the drop is not silent.
    p = _panel(30)
    with caplog.at_level(logging.INFO, logger="cbp.eval.walkforward"):
        out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    assert len(out) == 10  # behavior unchanged: 30 - 20 OOS predictions
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert infos, "expected an INFO log for the skipped leading releases"
    msg = " ".join(r.getMessage() for r in infos)
    assert "20" in msg  # states how many leading releases were skipped (n0)

def test_multivariate_recovers_two_feature_signal():
    rng = np.random.default_rng(3)
    n = 120
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    surprise = rng.normal(size=n)
    stance = rng.normal(size=n)
    target = 1.5 * surprise + 0.8 * stance + 0.05 * rng.normal(size=n)
    p = pd.DataFrame({"release_ts": ts, "surprise": surprise, "stance": stance, "DGS2_h1": target})
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20, feature_cols=["surprise", "stance"])
    corr = np.corrcoef(out["y_true"], out["y_pred"])[0, 1]
    assert corr > 0.95                                       # both features used -> strong OOS skill

def test_default_feature_cols_is_stance_only():
    # Backward compat: omitting feature_cols selects ["stance"] exactly as Phase 0.
    p = _panel(30)
    out = run_walkforward(p, "DGS2_h1", SimpleOLS(), ZeroChange(), n0=20)
    assert len(out) == 10
    assert list(out.columns) == ["release_ts", "y_true", "y_pred", "y_base"]
