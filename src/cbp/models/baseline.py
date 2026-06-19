# src/cbp/models/baseline.py
from __future__ import annotations
import numpy as np

class ZeroChange:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

class MeanModel:
    def __init__(self) -> None:
        self._mean: float = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)

class SimpleOLS:
    def __init__(self) -> None:
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        A = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self._intercept = float(beta[0]); self._coef = beta[1:]
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._intercept + X @ self._coef
