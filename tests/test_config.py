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
