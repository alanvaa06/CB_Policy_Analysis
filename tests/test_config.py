# tests/test_config.py
from pathlib import Path
from cbp.config import Config

def test_defaults():
    c = Config()
    assert c.horizons == (1, 5, 22)
    assert c.n0 == 20
    assert c.target_series == ("DGS2", "EFFR")
    assert c.event_window == (-1, 1)
    assert isinstance(c.data_dir, Path)

def test_frozen():
    c = Config()
    import pytest
    with pytest.raises(Exception):
        c.n0 = 5  # frozen
