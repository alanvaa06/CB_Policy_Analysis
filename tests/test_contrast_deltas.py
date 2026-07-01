# tests/test_contrast_deltas.py
import pandas as pd
from cbp.monitor.contrast import tone_deltas, all_pair_deltas


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01"]),
        "action": [1.0, -1.0, 0.0],
        "lexicon_tone": [0.5, 0.2, 0.0],
        "roberta_stance": [0.4, 0.1, float("nan")],
    })


def test_all_pair_deltas_keyed_by_latest_date():
    m = all_pair_deltas(_hist())
    assert set(m) == {"2024-03-20", "2024-05-01"}          # every date except the first


def test_all_pair_deltas_matches_tone_deltas_on_last_pair():
    hist = _hist()
    assert all_pair_deltas(hist)["2024-05-01"] == tone_deltas(hist)


def test_all_pair_deltas_none_delta_on_nan():
    m = all_pair_deltas(_hist())
    assert m["2024-05-01"]["roberta_stance"]["delta"] is None  # nan latest -> None


def test_all_pair_deltas_empty_when_single_row():
    one = _hist().iloc[:1]
    assert all_pair_deltas(one) == {}
