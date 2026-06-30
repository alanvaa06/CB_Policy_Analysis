# tests/test_action_lexicon.py
"""The descriptive action-tone lexicon (data/lexicons/action_tone.json) — a
monitoring-only index of the rate DECISION (raise/lower the target). Distinct
from the predictive stance lexicon; deliberately NOT subject to the size bounds
of hawk_dove.json."""
from pathlib import Path
from cbp.models.lexicon_scorer import load_lexicon, score_statement_lexicon

LEX = Path("data/lexicons/action_tone.json")


def test_action_lexicon_loads_disjoint_minimal():
    hawk, dove = load_lexicon(LEX)
    assert "raise" in hawk and "lower" in dove
    assert hawk.isdisjoint(dove)


def test_action_tone_directional_on_real_fomc_phrasing():
    hawk, dove = load_lexicon(LEX)
    cut = "the Committee decided to lower its target for the federal funds rate by 1/4 percentage point"
    hike = "the Committee voted to raise its target for the federal funds rate"
    hold = "the Committee decided to maintain the target range for the federal funds rate"
    assert score_statement_lexicon(cut, hawk, dove) == -1.0    # cut = dovish action
    assert score_statement_lexicon(hike, hawk, dove) == 1.0    # hike = hawkish action
    assert score_statement_lexicon(hold, hawk, dove) == 0.0    # hold = neutral
