# src/cbp/eval/walkforward.py
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def run_walkforward(panel: pd.DataFrame, target_col: str, model, baseline, n0: int, feature_cols=("stance",)) -> pd.DataFrame:
    """Expanding-window OOS. For each release i >= n0, train on rows [0, i) and
    predict row i. Training never sees row i or any later row -> no look-ahead.

    `feature_cols` selects the design-matrix columns (default ["stance"] for
    Phase 0 compatibility; Phase 1 nested comparison passes ["surprise"] and
    ["surprise", "stance"]).
    """
    df = panel.sort_values("release_ts").reset_index(drop=True)
    n_skipped = min(n0, len(df))
    logger.info(
        "Skipping first %d leading release(s) below the n0=%d minimum train size",
        n_skipped,
        n0,
    )
    y = df[target_col].to_numpy(dtype=float)
    X = df[list(feature_cols)].to_numpy(dtype=float)
    recs = []
    for i in range(n0, len(df)):
        Xtr, ytr = X[:i], y[:i]
        model.fit(Xtr, ytr); baseline.fit(Xtr, ytr)
        recs.append({
            "release_ts": df["release_ts"].iloc[i],
            "y_true": y[i],
            "y_pred": float(model.predict(X[i:i+1])[0]),
            "y_base": float(baseline.predict(X[i:i+1])[0]),
        })
    return pd.DataFrame(recs, columns=["release_ts", "y_true", "y_pred", "y_base"])
