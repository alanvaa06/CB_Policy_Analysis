# tests/test_stance.py
import pandas as pd
from cbp.data.fomc_calendar import load_fomc_calendar
from cbp.data.stance import load_stance

def test_stance_joins_release_ts(tmp_path):
    cal_p = tmp_path / "fomc_dates.csv"
    cal_p.write_text("release_date\n2020-01-29\n2020-03-18\n")
    cal = load_fomc_calendar(cal_p)
    st_p = tmp_path / "tdw_stance.csv"
    st_p.write_text("date,stance\n2020-01-29,0.5\n2020-03-18,-0.8\n")
    s = load_stance(st_p, cal)
    assert list(s.columns) == ["release_date", "release_ts", "stance", "doc_type"]
    assert s["stance"].tolist() == [0.5, -0.8]
    assert s["release_ts"].dt.tz is not None
    assert (s["doc_type"] == "statement").all()
