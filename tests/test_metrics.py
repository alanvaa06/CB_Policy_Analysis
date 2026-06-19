# tests/test_metrics.py
import numpy as np
from cbp.eval.metrics import oos_r2, rmse, hit_rate, sign_test

def test_rmse_zero_when_perfect():
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

def test_oos_r2_one_when_perfect():
    y = np.array([1.0, -2.0, 3.0]); base = np.zeros(3)
    assert abs(oos_r2(y, y, base) - 1.0) < 1e-12

def test_oos_r2_zero_when_equals_baseline():
    y = np.array([1.0, -2.0, 3.0]); base = np.full(3, 0.6667)
    assert abs(oos_r2(y, base, base)) < 1e-9

def test_hit_rate_direction():
    y = np.array([1.0, -1.0, 2.0, -3.0]); p = np.array([0.2, -0.1, -0.5, -1.0])
    assert hit_rate(y, p) == 0.75  # 3 of 4 signs match

def test_sign_test_keys():
    out = sign_test(np.array([1.0, -1.0]), np.array([0.5, -0.5]))
    assert {"hits", "n", "pvalue"} <= set(out)
