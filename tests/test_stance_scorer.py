from cbp.models.stance_scorer import split_sentences


def test_splits_on_terminal_punctuation():
    text = "The Committee raised rates. Inflation remains elevated! Will it persist?"
    assert split_sentences(text) == [
        "The Committee raised rates.",
        "Inflation remains elevated!",
        "Will it persist?",
    ]


def test_collapses_whitespace_and_newlines():
    text = "First sentence.\nSecond sentence.   Third sentence."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_empty_and_blank_return_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


import pandas as pd
import pytest
from cbp.models.stance_scorer import score_statements, LABEL_MAP


def _fake_classifier(texts: list[str]) -> list[dict]:
    # Deterministic stand-in for FOMC-RoBERTa: keyword -> label.
    out = []
    for t in texts:
        low = t.lower()
        if "hike" in low or "raise" in low:
            out.append({"label": "LABEL_1", "score": 0.9})   # Hawkish -> +1
        elif "cut" in low or "lower" in low:
            out.append({"label": "LABEL_0", "score": 0.9})   # Dovish -> -1
        else:
            out.append({"label": "LABEL_2", "score": 0.9})   # Neutral -> 0
    return out


def test_label_map_values():
    assert LABEL_MAP == {"LABEL_0": -1.0, "LABEL_1": 1.0, "LABEL_2": 0.0}


def test_score_is_mean_of_mapped_labels():
    # 2 hawkish + 1 neutral -> (1 + 1 + 0) / 3 = 0.6666...
    stmts = pd.DataFrame({
        "date": [pd.Timestamp("2008-01-30")],
        "text": ["We raise the rate. We will hike further. We monitor data."],
    })
    out = score_statements(stmts, _fake_classifier)
    assert list(out.columns) == ["date", "stance"]
    assert out["stance"].iloc[0] == pytest.approx(2.0 / 3.0)


def test_dovish_statement_scores_negative():
    stmts = pd.DataFrame({
        "date": [pd.Timestamp("2008-09-16")],
        "text": ["We cut the rate. We will lower further."],   # -1, -1 -> -1.0
    })
    out = score_statements(stmts, _fake_classifier)
    assert out["stance"].iloc[0] == pytest.approx(-1.0)
