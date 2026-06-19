# tests/test_mp_surprise.py
import pandas as pd
from cbp.data.mp_surprise import load_surprise


def test_load_surprise_reads_date_and_orthogonal_column(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({
        "date": ["2008-01-30", "2008-03-18", "2008-04-30"],
        "MP1_orthogonal": [0.05, -0.10, 0.02],
        "ignored_col": [9, 9, 9],
    }).to_excel(p, index=False)

    out = load_surprise(p)
    assert list(out.columns) == ["date", "surprise"]
    assert out["surprise"].tolist() == [0.05, -0.10, 0.02]
    assert out["date"].dt.tz is None                         # tz-naive calendar dates
    assert out["date"].iloc[0] == pd.Timestamp("2008-01-30")


def test_load_surprise_drops_missing_and_sorts(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({
        "date": ["2008-03-18", "2008-01-30"],                # out of order
        "MP1_orthogonal": [None, 0.05],                      # one missing -> dropped
    }).to_excel(p, index=False)

    out = load_surprise(p)
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2008-01-30")
    assert out["surprise"].iloc[0] == 0.05


def test_load_surprise_custom_column_names(tmp_path):
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({"meeting": ["2010-11-03"], "u_t": [0.123]}).to_excel(p, index=False)
    out = load_surprise(p, date_col="meeting", surprise_col="u_t")
    assert out["surprise"].iloc[0] == 0.123


def test_load_surprise_collapses_same_date_keeping_last(tmp_path):
    # The real BS file double-prints some pre-1999 unscheduled actions (two
    # intraday windows on one date). The per-meeting contract is one row per
    # date — same-date rows collapse to the LAST (later intraday window).
    p = tmp_path / "bs.xlsx"
    pd.DataFrame({
        "date": ["1991-12-20", "1991-12-20", "2008-01-30"],   # first date doubled
        "MP1_orthogonal": [-0.2401, 0.0523, 0.05],            # keep last (0.0523)
    }).to_excel(p, index=False)
    out = load_surprise(p)
    assert len(out) == 2                                       # one row per date
    assert out["date"].tolist() == [pd.Timestamp("1991-12-20"), pd.Timestamp("2008-01-30")]
    assert out["surprise"].iloc[0] == 0.0523                  # later intraday window kept
