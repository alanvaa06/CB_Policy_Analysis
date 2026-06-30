# tests/test_plot_tone.py
import pandas as pd
from cbp.viz.tone_compare import build_tone_comparison


def test_build_tone_comparison_inner_joins_on_date_and_correlates():
    lex = pd.DataFrame({"date": pd.to_datetime(["2020-01-29", "2020-03-15", "2020-04-29"]),
                        "stance": [1.0, -1.0, 0.5]})
    rob = pd.DataFrame({"date": pd.to_datetime(["2020-01-29", "2020-03-15"]),
                        "stance": [0.8, -0.6]})
    merged, corr = build_tone_comparison(lex, rob)
    assert list(merged.columns) == ["date", "stance_lexicon", "stance_roberta"]
    assert len(merged) == 2                       # inner join drops the unmatched date
    assert -1.0 <= corr <= 1.0
    assert corr > 0                               # both move the same direction here
