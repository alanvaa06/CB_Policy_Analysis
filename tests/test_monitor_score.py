# tests/test_monitor_score.py
import math
import pandas as pd
from cbp.monitor.score import score_all_measures
from cbp.monitor.history import HISTORY_COLUMNS


def _statements():
    # one clearly hawkish action ("raise"), one dovish-stance statement
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "text": ["The Committee decided to raise the target range. Policy is restrictive.",
                 "The Committee decided to lower rates. Policy is accommodative."],
    })


def fake_roberta(texts):
    # all sentences hawkish (+1) -> roberta_stance == 1.0 for every statement
    return [{"label": "LABEL_1"} for _ in texts]


def test_score_all_measures_columns_and_merge():
    out = score_all_measures(_statements(), lexicon_dir="data/lexicons", roberta=fake_roberta)
    assert list(out.columns) == HISTORY_COLUMNS
    assert len(out) == 2
    # n_sentences counted via split_sentences (2 sentences each)
    assert list(out["n_sentences"]) == [2, 2]
    # roberta injected -> all +1.0
    assert all(v == 1.0 for v in out["roberta_stance"])
    # action lexicon: raise -> +1, lower -> -1
    assert out.loc[out["date"] == pd.Timestamp("2024-01-31"), "action"].iloc[0] == 1.0
    assert out.loc[out["date"] == pd.Timestamp("2024-03-20"), "action"].iloc[0] == -1.0


def test_score_all_measures_no_roberta_is_nan():
    out = score_all_measures(_statements(), lexicon_dir="data/lexicons", roberta=None)
    assert out["roberta_stance"].isna().all()
    # the light measures still populate
    assert not out["action"].isna().any()
