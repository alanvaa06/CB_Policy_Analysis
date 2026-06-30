# src/cbp/viz/tone_compare.py
from __future__ import annotations

import pandas as pd


def build_tone_comparison(lex: pd.DataFrame, rob: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Inner-join lexicon and RoBERTa stance on date; return (merged, pearson_corr).
    merged columns: [date, stance_lexicon, stance_roberta]."""
    m = (lex.rename(columns={"stance": "stance_lexicon"})
            .merge(rob.rename(columns={"stance": "stance_roberta"}), on="date", how="inner")
            .sort_values("date").reset_index(drop=True))
    corr = float(m["stance_lexicon"].corr(m["stance_roberta"]))
    return m[["date", "stance_lexicon", "stance_roberta"]], corr
