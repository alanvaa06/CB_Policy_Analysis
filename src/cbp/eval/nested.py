# src/cbp/eval/nested.py
from __future__ import annotations
import pandas as pd
import statsmodels.api as sm
from cbp.models.baseline import SimpleOLS, ZeroChange
from cbp.eval.walkforward import run_walkforward
from cbp.eval.metrics import oos_r2


def _stance_partial_t(panel: pd.DataFrame, target_col: str) -> float:
    """In-sample OLS target ~ const + surprise + stance over the full panel;
    return the t-stat of the stance coefficient. Descriptive companion to the OOS
    delta_r2 (PRD §7), NOT itself an OOS metric.
    """
    df = panel.dropna(subset=[target_col, "surprise", "stance"])
    X = sm.add_constant(df[["surprise", "stance"]].to_numpy(dtype=float))  # [const, surprise, stance]
    res = sm.OLS(df[target_col].to_numpy(dtype=float), X).fit()
    return float(res.tvalues[2])


def residual_stance_regression(panel: pd.DataFrame, target_col: str, n0: int) -> dict:
    """Cross-check (PRD §7): regress the surprise-only OOS residual on stance.

    residual = y_true - (surprise-only OOS fit). A hawkish-text -> higher-residual
    -yield read independent of the decision. Returns slope/t/r2/n; degrades to NaN
    if fewer than 2 points or stance is constant.
    """
    a = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise"])
    df = panel.sort_values("release_ts").reset_index(drop=True)
    merged = a.merge(df[["release_ts", "stance"]], on="release_ts", how="left")
    resid = (merged["y_true"] - merged["y_pred"]).to_numpy(dtype=float)
    stance = merged["stance"].to_numpy(dtype=float)
    if len(resid) < 2 or len(set(stance)) < 2:
        return {"slope": float("nan"), "tstat": float("nan"), "r2": float("nan"), "n": int(len(resid))}
    res = sm.OLS(resid, sm.add_constant(stance)).fit()
    return {"slope": float(res.params[1]), "tstat": float(res.tvalues[1]), "r2": float(res.rsquared), "n": int(len(resid))}


def nested_oos(panel: pd.DataFrame, target_col: str, n0: int) -> dict:
    """Nested OOS comparison: surprise-only (A) vs surprise+stance (B).

    Runs the Phase 0 walk-forward twice on the same panel and n0 against the same
    ZeroChange baseline; delta_r2 = oos_r2(B) - oos_r2(A). The statement text
    carries marginal predictive information iff delta_r2 > 0.
    """
    a = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise"])
    b = run_walkforward(panel, target_col, SimpleOLS(), ZeroChange(), n0, feature_cols=["surprise", "stance"])
    r2_base = oos_r2(a["y_true"].to_numpy(), a["y_pred"].to_numpy(), a["y_base"].to_numpy())
    r2_full = oos_r2(b["y_true"].to_numpy(), b["y_pred"].to_numpy(), b["y_base"].to_numpy())
    return {
        "r2_base": r2_base,
        "r2_full": r2_full,
        "delta_r2": r2_full - r2_base,
        "n": int(len(b)),
        "stance_partial_t": _stance_partial_t(panel, target_col),
    }
