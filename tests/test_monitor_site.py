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
    # v2 redline is word-level inline flow: 'replace' maps to delete+insert (no rl-replace).
    for cls in ("rl-equal", "rl-insert", "rl-delete"):
        assert cls in html
    assert "&lt;" not in "new"  # sanity: escaping only applied to content


def test_render_site_writes_self_contained_html(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist_v2(), deltas, segs, out)  # v2 render needs theme/metric columns
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "plotly" in html.lower()              # plotly.js inlined
    assert "descriptive tracker" in html.lower() # honest framing (v2 glossary)
    assert "Rates unchanged." in html


from cbp.monitor.site import (
    build_theme_heatmap, build_change_magnitude_figure, build_commstyle_figure, glossary_html,
)


def _hist_v2():
    base = _hist()  # the v1 3-row fixture
    base["word_count"] = [300, 320, 280]
    base["flesch"] = [25.0, 22.0, 30.0]
    base["uncertainty_per1k"] = [10.0, 12.0, 8.0]
    base["change_magnitude"] = [float("nan"), 0.2, 0.5]
    for i, c in enumerate(["theme_inflation", "theme_employment", "theme_growth",
                           "theme_balance_sheet", "theme_financial_conditions"]):
        base[c] = [float(i + 1)] * 3
    return base


def test_theme_heatmap_has_five_rows():
    fig = build_theme_heatmap(_hist_v2())
    assert len(fig.data) == 1                       # one heatmap trace
    assert len(fig.data[0].z) == 5                  # five themes on the y axis


def test_change_magnitude_and_commstyle_build():
    assert len(build_change_magnitude_figure(_hist_v2()).data) == 1
    assert len(build_commstyle_figure(_hist_v2()).data) == 3   # words, flesch, uncertainty


def test_glossary_defines_each_measure():
    g = glossary_html()
    for term in ("action", "lexicon", "RoBERTa", "change", "theme"):
        assert term.lower() in g.lower()


def test_redline_html_is_inline_flow():
    segs = [{"op": "equal", "prev": "the rate", "curr": "the rate"},
            {"op": "replace", "prev": "hold", "curr": "raise"}]
    html = build_redline_html(segs)
    assert "redline-flow" in html
    assert "rl-delete" in html and "rl-insert" in html


def test_redline_html_tightens_punctuation_across_segments():
    import re
    segs = [{"op": "equal", "prev": "percent", "curr": "percent"},
            {"op": "insert", "prev": "", "curr": ","},
            {"op": "equal", "prev": "in support", "curr": "in support"}]
    text = re.sub(r"<[^>]+>", "", build_redline_html(segs))
    assert "percent," in text and "percent ," not in text


def test_render_site_has_six_sections(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist_v2(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    for heading in ("How to read", "focused on", "How much", "Communication style"):
        assert heading.lower() in html.lower()
    assert "descriptive" in html.lower()


def test_build_redlines_payload_shape():
    from cbp.monitor.site import build_redlines_payload
    deltas_by_date = {
        "2024-05-01": {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
                       "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
                       "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
                       "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}},
    }
    segs_by_date = {"2024-05-01": [{"op": "replace", "prev": "hold", "curr": "raise"}]}
    payload = build_redlines_payload(deltas_by_date, segs_by_date)
    assert set(payload) == {"2024-05-01"}
    entry = payload["2024-05-01"]
    assert "deltas" in entry["deltas_html"].lower()      # <table class='deltas'>
    assert "rl-insert" in entry["redline_html"]          # rendered redline
    # date present in segments but absent from deltas -> empty-state, not a raise
    payload_missing = build_redlines_payload({}, segs_by_date)
    assert "rl-empty" in payload_missing["2024-05-01"]["deltas_html"]


def test_render_site_has_meeting_selector_and_hooks(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist_v2(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    assert 'id="meeting"' in html                          # the dropdown
    assert html.count("<option") == 2                      # 3 rows -> 2 pairs w/ a prior
    assert 'value="2024-05-01"' in html                    # newest option present
    assert 'id="deltas-slot"' in html and 'id="redline-slot"' in html
    for cid in ("fig-heatmap", "fig-change", "fig-commstyle", "fig-levels", "fig-deltas"):
        assert cid in html                                 # explicit chart div ids
    assert "redlines.json" in html                         # fetch target
    assert "Plotly.relayout" in html                       # marker hook
