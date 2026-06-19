# tests/test_fred.py
import numpy as np
from cbp.data.fred import parse_fred_observations

def test_parse_handles_missing_dot():
    obs = [
        {"date": "2020-01-02", "value": "1.57"},
        {"date": "2020-01-03", "value": "."},     # FRED missing marker
        {"date": "2020-01-06", "value": "1.60"},
    ]
    s = parse_fred_observations(obs, "DGS2")
    assert s.name == "DGS2"
    assert s.loc["2020-01-02"] == 1.57
    assert np.isnan(s.loc["2020-01-03"])
    assert s.index.is_monotonic_increasing
