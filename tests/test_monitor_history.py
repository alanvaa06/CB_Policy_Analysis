# tests/test_monitor_history.py
import pandas as pd
from cbp.monitor.history import (
    HISTORY_COLUMNS, load_history, upsert_history, save_history,
)


def _row(date, action=0.0, lex=0.0, rob=0.0, n=5):
    return {"date": pd.Timestamp(date), "action": action, "lexicon_tone": lex,
            "roberta_stance": rob, "n_sentences": n}


def test_load_history_missing_returns_empty_schema(tmp_path):
    h = load_history(tmp_path / "nope.csv")
    assert list(h.columns) == HISTORY_COLUMNS
    assert len(h) == 0


def test_upsert_appends_and_sorts(tmp_path):
    h = pd.DataFrame([_row("2024-03-20", action=-1.0)])
    new = pd.DataFrame([_row("2024-01-31", action=1.0)])
    out = upsert_history(h, new)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-03-20"]


def test_upsert_same_date_overwrites(tmp_path):
    h = pd.DataFrame([_row("2024-03-20", action=-1.0, rob=-0.5)])
    new = pd.DataFrame([_row("2024-03-20", action=-1.0, rob=-0.9)])
    out = upsert_history(h, new)
    assert len(out) == 1
    assert out.loc[0, "roberta_stance"] == -0.9   # new wins


def test_save_then_load_roundtrip_is_idempotent(tmp_path):
    p = tmp_path / "hist.csv"
    h = upsert_history(load_history(p), pd.DataFrame([_row("2024-01-31", action=1.0, rob=0.25)]))
    save_history(h, p)
    reloaded = load_history(p)
    assert reloaded.loc[0, "action"] == 1.0
    assert reloaded.loc[0, "roberta_stance"] == 0.25
    # re-upserting the same row is a no-op on length
    again = upsert_history(reloaded, pd.DataFrame([_row("2024-01-31", action=1.0, rob=0.25)]))
    assert len(again) == 1


from cbp.monitor.history import METRIC_COLUMNS, THEME_COLUMNS


def test_history_columns_include_metric_and_theme_columns():
    assert THEME_COLUMNS == ["theme_inflation", "theme_employment", "theme_growth",
                             "theme_balance_sheet", "theme_financial_conditions"]
    for c in ["word_count", "flesch", "uncertainty_per1k", "change_magnitude", *THEME_COLUMNS]:
        assert c in METRIC_COLUMNS
        assert c in HISTORY_COLUMNS


def test_extended_row_roundtrips(tmp_path):
    p = tmp_path / "hist.csv"
    row = {c: 0.0 for c in HISTORY_COLUMNS}
    row["date"] = pd.Timestamp("2024-01-31")
    row["theme_inflation"] = 12.5
    row["change_magnitude"] = 0.4
    save_history(pd.DataFrame([row]), p)
    back = load_history(p)
    assert back.loc[0, "theme_inflation"] == 12.5
    assert back.loc[0, "change_magnitude"] == 0.4
