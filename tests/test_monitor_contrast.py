# tests/test_monitor_contrast.py
import math
import pandas as pd
from cbp.monitor.contrast import MEASURES, tone_deltas, redline


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "action": [1.0, -1.0],
        "lexicon_tone": [0.5, 0.2],
        "roberta_stance": [0.4, float("nan")],
        "n_sentences": [10, 12],
    })


def test_tone_deltas_basic_and_nan():
    d = tone_deltas(_hist())
    assert d["date_latest"] == "2024-03-20" and d["date_prior"] == "2024-01-31"
    assert d["action"]["delta"] == -2.0
    assert math.isclose(d["lexicon_tone"]["delta"], -0.3, abs_tol=1e-9)
    assert d["roberta_stance"]["delta"] is None   # latest is NaN


def test_tone_deltas_needs_two_rows():
    one = _hist().iloc[:1]
    assert tone_deltas(one) == {}


def test_redline_detects_equal_insert_delete_replace():
    prev = "Rates unchanged. Inflation remains elevated. Risks are balanced."
    curr = "Rates unchanged. Inflation has eased. Risks are balanced. New paragraph."
    segs = redline(prev, curr)
    ops = [s["op"] for s in segs]
    assert "equal" in ops      # "Rates unchanged."
    assert "replace" in ops    # inflation sentence reworded
    assert "insert" in ops     # "New paragraph."
    # a pure deletion case
    segs2 = redline("A sentence. B sentence.", "A sentence.")
    assert any(s["op"] == "delete" and "B sentence" in s["prev"] for s in segs2)
