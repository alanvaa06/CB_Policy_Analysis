# tests/test_monitor_metrics.py
import json
from pathlib import Path

import pytest

from cbp.monitor.metrics import load_themes


def test_load_themes_returns_lowercase_frozensets(tmp_path):
    p = tmp_path / "themes.json"
    p.write_text(json.dumps({
        "themes": {"inflation": ["Inflat", "PRICE"], "growth": ["growth"]},
        "uncertainty": ["Uncertain", "risk"],
    }))
    themes, unc = load_themes(p)
    assert themes["inflation"] == frozenset({"inflat", "price"})
    assert themes["growth"] == frozenset({"growth"})
    assert unc == frozenset({"uncertain", "risk"})


def test_load_themes_missing_raises_valueerror():
    with pytest.raises(ValueError, match="themes"):
        load_themes(Path("does/not/exist.json"))


def test_repo_themes_file_has_five_themes_and_uncertainty():
    themes, unc = load_themes(Path("data/lexicons/themes.json"))
    assert set(themes) == {"inflation", "employment", "growth",
                           "balance_sheet", "financial_conditions"}
    assert all(len(v) >= 3 for v in themes.values())
    assert len(unc) >= 3


import math

from cbp.monitor.metrics import (
    clean_statement, word_count, flesch, count_themes, uncertainty_count, change_magnitude,
)

_MODERN = ("April 29, 2026 For release at 2:00 p.m. EDT Share Recent indicators suggest "
           "that economic activity has been expanding at a solid pace. Inflation is elevated. "
           "Voting for the monetary policy action were Jerome H. Powell and others. "
           "For media inquiries, please email someone or call 202-452-2955.")


def test_clean_statement_strips_header_voting_and_media():
    c = clean_statement(_MODERN)
    assert c.startswith("Recent indicators")          # release header + "Share" gone
    assert "Voting for" not in c                        # voting roster gone
    assert "media inquiries" not in c                   # media line gone
    assert "Inflation is elevated." in c                # substance kept


def test_clean_statement_passthrough_when_no_boilerplate():
    raw = "The Committee decided to maintain the target range."
    assert clean_statement(raw) == raw


def test_clean_statement_never_empty():
    assert clean_statement("   ") == ""  # strip of blank -> "" via fallback to stripped raw


def test_word_count_counts_alphabetic_tokens():
    assert word_count("Rates at 5.25% will hold.") == 4  # rates, at, will, hold


def test_flesch_is_higher_for_simpler_text():
    simple = "The cat sat. The dog ran."
    complex_ = ("Notwithstanding heterogeneous macroprudential considerations, the Committee "
                "reaffirmed its accommodative configuration.")
    assert flesch(simple) > flesch(complex_)


def test_flesch_empty_is_zero():
    assert flesch("   ") == 0.0


def test_count_themes_prefix_matches():
    themes = {"inflation": frozenset({"inflat", "price"}), "growth": frozenset({"growth"})}
    counts = count_themes("Inflation and prices rose; growth slowed and inflationary risk", themes)
    assert counts["inflation"] == 3   # Inflation, prices, inflationary
    assert counts["growth"] == 1


def test_uncertainty_count():
    assert uncertainty_count("risks remain and the outcome is uncertain",
                             frozenset({"risk", "uncertain"})) == 2


def test_change_magnitude_bounds():
    assert change_magnitude("a b c d", "a b c d") == 0.0           # identical
    assert change_magnitude("a b c", "x y z") == 1.0               # disjoint
    mid = change_magnitude("the committee will hold rates steady",
                           "the committee will raise rates sharply")
    assert 0.0 < mid < 1.0


def test_clean_statement_repairs_mojibake():
    # en-dash bytes E2 80 93 mis-decoded as Latin-1 -> U+00E2 U+0080 U+0093
    fixed = clean_statement("approved by a 12 â 0 vote")
    assert "–" in fixed                    # proper en dash restored
    assert "â" not in fixed                 # mojibake marker gone
