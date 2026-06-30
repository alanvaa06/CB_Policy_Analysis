# tests/test_cli.py
import pytest
import numpy as np
import pandas as pd
from cbp.config import Config
from cbp.cli import run_report

def test_run_report_offline():
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rng = np.random.default_rng(0)
    stance_by_release = {}
    # ~8 releases/yr; build a market that rises after hawkish stance
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    stance = rng.normal(size=len(rel_ts))
    market = pd.DataFrame(index=idx)
    base = np.zeros(len(idx))
    for ts, s in zip(rel_ts, stance):
        base[idx > ts] += 0.02 * s        # hawkish -> drift up after release
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))
    market["EFFR"] = market["DGS2"]
    stance_df = pd.DataFrame({"release_ts": rel_ts, "stance": stance, "doc_type": "statement",
                              "release_date": rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)})
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))
    report = run_report(market, stance_df, cfg)
    assert ("DGS2", 1) in report["oos"]
    assert report["oos"][("DGS2", 1)]["n"] > 0
    assert np.isfinite(report["oos"][("DGS2", 1)]["oos_r2"])


from cbp.cli import run_nested_report

def test_run_nested_report_offline():
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rng = np.random.default_rng(0)
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    stance = rng.normal(size=len(rel_ts))
    surprise = rng.normal(size=len(rel_ts))
    market = pd.DataFrame(index=idx)
    base = np.zeros(len(idx))
    for ts, s, u in zip(rel_ts, stance, surprise):
        base[idx > ts] += 0.02 * s + 0.03 * u              # post-release drift = stance + surprise
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))

    rel_date = rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)
    stance_df = pd.DataFrame({"release_ts": rel_ts, "stance": stance, "doc_type": "statement", "release_date": rel_date})
    surprise_df = pd.DataFrame({"date": rel_date, "surprise": surprise})
    cfg = Config(horizons=(1, 5), target_series=("DGS2",))

    report = run_nested_report(market, stance_df, surprise_df, cfg)
    assert ("DGS2", 1) in report["nested"]
    assert ("DGS2", 1) in report["residual"]
    nested = report["nested"][("DGS2", 1)]
    assert set(nested) == {"r2_base", "r2_full", "delta_r2", "n", "stance_partial_t"}
    assert nested["n"] > 0
    assert np.isfinite(nested["delta_r2"])


from cbp.models.lexicon_scorer import score_statements_lexicon, load_lexicon
from cbp.data.stance import stance_frame_from_scores
from pathlib import Path


def test_lexicon_stance_runs_through_nested_report_offline():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    rel_date = rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)
    # alternate hawkish/dovish statement text so lexicon stance is non-degenerate
    texts = ["tighten restrictive tightening" if i % 2 else "accommodative easing stimulus"
             for i in range(len(rel_ts))]
    statements = pd.DataFrame({"date": rel_date, "text": texts})
    hawk, dove = load_lexicon(Path("data/lexicons/hawk_dove.json"))
    scores = score_statements_lexicon(statements, hawk, dove)
    cal = pd.DataFrame({"release_date": rel_date, "release_ts": rel_ts})
    stance_df = stance_frame_from_scores(scores, cal)
    assert stance_df["stance"].abs().sum() > 0          # non-degenerate

    surprise = rng.normal(size=len(rel_ts))
    base = np.zeros(len(idx))
    for ts, u in zip(rel_ts, surprise):
        base[idx > ts] += 0.03 * u
    market = pd.DataFrame(index=idx)
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))
    surprise_df = pd.DataFrame({"date": rel_date, "surprise": surprise})
    cfg = Config(horizons=(1,), target_series=("DGS2",))
    report = run_nested_report(market, stance_df, surprise_df, cfg)
    assert np.isfinite(report["nested"][("DGS2", 1)]["delta_r2"])


def test_cli_argparser_tone_method_default_and_choices():
    import argparse
    from cbp.cli import build_parser
    p = build_parser()
    assert p.parse_args([]).tone_method == "roberta"
    assert p.parse_args(["--tone-method", "lexicon"]).tone_method == "lexicon"
    with pytest.raises(SystemExit):
        p.parse_args(["--tone-method", "bogus"])
