# tests/test_stance.py
import pandas as pd
from cbp.data.fomc_calendar import load_fomc_calendar
from cbp.data.stance import load_stance, stance_frame_from_scores

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

def test_stance_frame_from_scores_joins_calendar(tmp_path):
    cal_p = tmp_path / "fomc_dates.csv"
    cal_p.write_text("release_date\n2020-01-29\n2020-03-18\n")
    cal = load_fomc_calendar(cal_p)
    scores = pd.DataFrame({
        "date": [pd.Timestamp("2020-03-18"), pd.Timestamp("2020-01-29")],  # out of order
        "stance": [-0.8, 0.5],
    })
    out = stance_frame_from_scores(scores, cal)
    assert list(out.columns) == ["release_date", "release_ts", "stance", "doc_type"]
    assert out["stance"].tolist() == [0.5, -0.8]            # sorted by release_ts
    assert out["release_ts"].dt.tz is not None
    assert (out["doc_type"] == "statement").all()
