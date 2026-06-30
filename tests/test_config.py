# tests/test_config.py
from pathlib import Path
from cbp.config import Config

def test_defaults():
    c = Config()
    assert c.horizons == (1, 5, 22)
    assert c.n0 == 20
    assert c.target_series == ("DGS2", "DGS1")     # EFFR dropped (mechanical), DGS1 added
    assert c.event_window == (-1, 1)
    assert isinstance(c.data_dir, Path)
    assert c.roberta_model_id == "gtfintechlab/FOMC-RoBERTa"

def test_frozen():
    c = Config()
    import pytest
    with pytest.raises(Exception):
        c.n0 = 5  # frozen


def test_config_has_lexicon_path_default():
    cfg = Config()
    assert cfg.lexicon_path == Path("data/lexicons/hawk_dove.json")


def test_config_monitor_paths_defaults():
    c = Config()
    assert c.monitor_dir == Path("data/monitor")
    assert c.history_path == Path("data/monitor/tone_history.csv")
    assert c.calendar_path == Path("data/monitor/fomc_calendar.csv")
    assert c.redline_path == Path("data/monitor/latest_redline.json")
    assert c.lexicon_dir == Path("data/lexicons")
    assert c.statements_dir == Path("data/raw/statements")
    assert c.site_out == Path("site/index.html")
