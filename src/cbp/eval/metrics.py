# src/cbp/eval/metrics.py
from __future__ import annotations
import numpy as np
from scipy import stats  # scipy ships with statsmodels; add to deps if missing

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    sse = np.sum((y_true - y_pred) ** 2)
    sse_base = np.sum((y_true - y_base) ** 2)
    return float(1.0 - sse / sse_base) if sse_base > 0 else float("nan")

def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def sign_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    hits = int(np.sum(np.sign(y_true) == np.sign(y_pred)))
    n = int(len(y_true))
    pvalue = float(stats.binomtest(hits, n, 0.5).pvalue) if n else float("nan")
    return {"hits": hits, "n": n, "pvalue": pvalue}
