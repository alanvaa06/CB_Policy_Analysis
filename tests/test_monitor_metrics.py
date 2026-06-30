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
