# tests/test_lexicon_scorer.py
import json
import logging
import pytest
import pandas as pd
from pathlib import Path
from cbp.models.lexicon_scorer import load_lexicon
from cbp.models.lexicon_scorer import tokenize
from cbp.models.lexicon_scorer import score_statement_lexicon
from cbp.models.lexicon_scorer import score_statements_lexicon


def test_load_lexicon_returns_two_nonempty_lowercase_frozensets(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"hawk": ["Tightening", "RESTRICTIVE"], "dove": ["accommodative"],
                             "sources": ["x"], "notes": "y"}))
    hawk, dove = load_lexicon(p)
    assert isinstance(hawk, frozenset) and isinstance(dove, frozenset)
    assert hawk == {"tightening", "restrictive"}   # lowercased
    assert dove == {"accommodative"}


def test_load_lexicon_missing_file_raises_valueerror():
    with pytest.raises(ValueError, match="lexicon"):
        load_lexicon(Path("does/not/exist.json"))


def test_load_lexicon_nonstring_entry_raises_valueerror(tmp_path):
    # malformed: list contains a non-string -> .lower() would AttributeError;
    # must surface as the promised ValueError naming the path (spec 003 §8)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"hawk": [1, 2], "dove": ["ease"]}))
    with pytest.raises(ValueError, match="lexicon"):
        load_lexicon(p)


def test_repo_lexicon_is_small_disjoint_and_policy_stance():
    hawk, dove = load_lexicon(Path("data/lexicons/hawk_dove.json"))
    assert 3 <= len(hawk) <= 8 and 3 <= len(dove) <= 8   # small, corpus-validated set
    assert hawk.isdisjoint(dove)
    # confound words MUST be excluded (verifier finding): boilerplate + condition valence + dead seeds
    banned = {"inflation", "weak", "downside", "robust", "slack", "upside",
              "gradual", "patient", "elevated", "hawkish", "dovish", "vigilan"}
    assert banned.isdisjoint(hawk | dove)
    # polarity-stable adjective chosen over flip-prone noun stem
    assert "accommodative" in dove and "accommodat" not in dove


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("The Committee will TIGHTEN, gradually.") == \
        ["the", "committee", "will", "tighten", "gradually"]


def test_tokenize_drops_digits_and_empty():
    assert tokenize("Rate at 5.25% — easing?") == ["rate", "at", "easing"]


def test_tokenize_empty_returns_empty_list():
    assert tokenize("   ") == []


HAWK = frozenset({"tighten", "restrictive"})
DOVE = frozenset({"accommodative", "easing"})


def test_net_tone_all_hawk_is_plus_one():
    assert score_statement_lexicon("tighten restrictive tightening", HAWK, DOVE) == 1.0


def test_net_tone_all_dove_is_minus_one():
    assert score_statement_lexicon("accommodative easing", HAWK, DOVE) == -1.0


def test_net_tone_balanced_is_zero():
    # 2 hawk (tighten, restrictive), 2 dove (accommodative, easing)
    assert score_statement_lexicon("tighten restrictive accommodative easing", HAWK, DOVE) == 0.0


def test_net_tone_no_keywords_is_zero_not_nan():
    assert score_statement_lexicon("the committee met today", HAWK, DOVE) == 0.0


def test_stem_prefix_matches_inflection():
    # "tightened" must match stem "tighten"
    assert score_statement_lexicon("tightened", HAWK, DOVE) == 1.0


def test_polarity_stable_adjective_not_flip_prone_noun():
    # "accommodative" stem must NOT match the flip-prone noun "accommodation"
    # ("removing accommodation" is hawkish) -> 0.0, not -1.0
    assert score_statement_lexicon("accommodation", HAWK, DOVE) == 0.0


def test_score_statements_lexicon_shape_and_values():
    statements = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-29", "2020-03-15"]),
        "text": ["tighten restrictive", "accommodative easing"],
    })
    out = score_statements_lexicon(statements, HAWK, DOVE)
    assert list(out.columns) == ["date", "stance"]
    assert len(out) == 2
    assert out.loc[0, "stance"] == 1.0
    assert out.loc[1, "stance"] == -1.0


def test_score_statements_lexicon_empty_text_scores_zero(caplog):
    statements = pd.DataFrame({"date": pd.to_datetime(["2020-01-29"]), "text": ["   "]})
    with caplog.at_level(logging.INFO):
        out = score_statements_lexicon(statements, HAWK, DOVE)
    assert len(out) == 1               # not dropped: empty text is a valid 0.0 reading
    assert out.loc[0, "stance"] == 0.0
    assert "scored neutral 0.0" in caplog.text
