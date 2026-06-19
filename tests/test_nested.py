# tests/test_nested.py
import numpy as np
import pandas as pd
from cbp.eval.nested import nested_oos

def _nested_panel(n, add_stance_signal: bool, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2000-01-01", periods=n, freq="W", tz="UTC")
    surprise = rng.normal(size=n)
    stance = rng.normal(size=n)
    coef_stance = 2.0 if add_stance_signal else 0.0
    target = 1.0 * surprise + coef_stance * stance + 0.05 * rng.normal(size=n)
    return pd.DataFrame({"release_ts": ts, "surprise": surprise, "stance": stance, "DGS2_h1": target})

def test_keys_present():
    out = nested_oos(_nested_panel(150, True), "DGS2_h1", n0=20)
    assert set(out) == {"r2_base", "r2_full", "delta_r2", "n", "stance_partial_t"}
    assert out["n"] == 130                                   # 150 - 20 OOS rows

def test_stance_adds_value_delta_r2_positive():
    out = nested_oos(_nested_panel(150, add_stance_signal=True), "DGS2_h1", n0=20)
    assert out["delta_r2"] > 0.05                            # full model beats surprise-only OOS
    assert out["stance_partial_t"] > 3.0                    # in-sample stance coef clearly nonzero

def test_null_stance_adds_nothing_delta_r2_near_zero():
    out = nested_oos(_nested_panel(150, add_stance_signal=False), "DGS2_h1", n0=20)
    assert out["delta_r2"] < 0.02                            # no improvement (small/<=0)
    assert abs(out["stance_partial_t"]) < 2.5                # stance coef not significant
