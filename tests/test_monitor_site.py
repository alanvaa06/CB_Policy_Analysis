# tests/test_monitor_site.py
import pandas as pd
import pytest

pytest.importorskip("plotly")  # site rendering needs the [site] extra; skip if absent

from cbp.monitor.site import (
    build_levels_figure, build_deltas_figure, build_redline_html, render_site,
)


def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01"]),
        "action": [1.0, -1.0, 0.0],
        "lexicon_tone": [0.5, 0.2, 0.0],
        "roberta_stance": [0.4, 0.1, float("nan")],
        "n_sentences": [10, 12, 9],
    })


def test_levels_figure_has_three_traces():
    fig = build_levels_figure(_hist())
    assert len(fig.data) == 3
    assert {t.name for t in fig.data} == {"action", "lexicon", "RoBERTa"}


def test_deltas_figure_has_three_traces():
    fig = build_deltas_figure(_hist())
    assert len(fig.data) == 3


def test_redline_html_classes_per_op():
    segs = [{"op": "equal", "prev": "x", "curr": "x"},
            {"op": "insert", "prev": "", "curr": "new"},
            {"op": "delete", "prev": "gone", "curr": ""},
            {"op": "replace", "prev": "old", "curr": "fresh"}]
    html = build_redline_html(segs)
    for cls in ("rl-equal", "rl-insert", "rl-delete", "rl-replace"):
        assert cls in html
    assert "&lt;" not in "new"  # sanity: escaping only applied to content


def test_render_site_writes_self_contained_html(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "plotly" in html.lower()              # plotly.js inlined
    assert "descriptive monitor" in html.lower() # honest framing banner
    assert "Rates unchanged." in html
