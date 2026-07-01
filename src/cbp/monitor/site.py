# src/cbp/monitor/site.py
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import pandas as pd

VERDICT_URL = "https://github.com/alanvaa06/CB_Policy_Analysis/blob/main/docs/results/2026-06-29-lexicon-baseline-verdict.md"
_LEVELS = [("action", "action"), ("lexicon_tone", "lexicon"), ("roberta_stance", "RoBERTa")]
_THEMES = [("theme_inflation", "inflation"), ("theme_employment", "employment"),
           ("theme_growth", "growth"), ("theme_balance_sheet", "balance sheet"),
           ("theme_financial_conditions", "financial")]


def build_levels_figure(history: pd.DataFrame):
    """Stance measures through time (kept, now in the 'stance in context' panel)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col], name=name,
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="Stance measures — levels (1999→latest)",
                      yaxis_title="stance", template="plotly_white", height=360)
    return fig


def build_deltas_figure(history: pd.DataFrame):
    """Meeting-over-meeting Δ per stance measure."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col].diff(), name=f"Δ {name}",
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="Stance measures — meeting-over-meeting change",
                      yaxis_title="Δ stance", template="plotly_white", height=360)
    return fig


def build_theme_heatmap(history: pd.DataFrame):
    """What the Fed is focused on: theme intensity (hits per 1,000 words) over time."""
    import plotly.graph_objects as go
    z = [history[col].tolist() for col, _ in _THEMES]
    fig = go.Figure(go.Heatmap(z=z, x=history["date"], y=[name for _, name in _THEMES],
                               colorscale="YlOrRd", colorbar={"title": "per 1k"}))
    fig.update_layout(title="What the Fed is focused on — theme intensity (per 1,000 words)",
                      template="plotly_white", height=320)
    return fig


def build_change_magnitude_figure(history: pd.DataFrame):
    """How much each statement was rewritten vs the prior (0=identical, 1=rewritten)."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=history["date"], y=history["change_magnitude"],
                               mode="lines+markers", name="change", connectgaps=False))
    fig.update_layout(title="How much each statement changed vs the prior",
                      yaxis_title="edit fraction (0–1)", template="plotly_white", height=320)
    return fig


def build_commstyle_figure(history: pd.DataFrame):
    """Communication style: length, readability, uncertainty as stacked sub-plots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("length (words)", "readability (Flesch)",
                                        "uncertainty (per 1,000 words)"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["word_count"], name="words"), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=history["flesch"], name="flesch"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=history["uncertainty_per1k"], name="uncertainty"), row=3, col=1)
    fig.update_layout(title="Communication style over time", template="plotly_white",
                      height=520, showlegend=False)
    return fig


_TIGHT_BEFORE = ".,;:%!?)]}"   # no space before a fragment starting with these
_TIGHT_AFTER = "([{$-/'"        # no space after a fragment ending with these


def _sep(prev_text: str, next_text: str) -> str:
    """Space between two redline fragments unless punctuation/fraction/possessive
    makes it tight (so `percent ,` -> `percent,`, `3-` + `1/2` -> `3-1/2`)."""
    if not prev_text:
        return ""
    a, b = prev_text[-1], next_text[:1]
    return "" if (b in _TIGHT_BEFORE or a in _TIGHT_AFTER) else " "


def build_redline_html(segments: list[dict]) -> str:
    """Inline word-level redline: one flowing paragraph, only changes highlighted.
    Fragments are joined with punctuation-aware spacing so the prose reads naturally
    even where a change boundary lands on punctuation."""
    if not segments:
        return '<p class="rl-empty">Need ≥2 statements to show a redline.</p>'
    units: list[tuple[str, str]] = []  # (css_class, visible_text)
    for s in segments:
        op = s["op"]
        if op == "equal":
            units.append(("rl-equal", s["curr"]))
        elif op == "insert":
            units.append(("rl-insert", s["curr"]))
        elif op == "delete":
            units.append(("rl-delete", s["prev"]))
        else:  # replace -> struck old then new
            units.append(("rl-delete", s["prev"]))
            units.append(("rl-insert", s["curr"]))
    out, last = [], ""
    for cls, text in units:
        if not text:
            continue
        out.append(f'{_sep(last, text)}<span class="{cls}">{_html.escape(text)}</span>')
        last = text
    return '<div class="redline-flow">' + "".join(out) + "</div>"


def glossary_html() -> str:
    """Plain-English definition of every measure on the page."""
    items = [
        ("action", "the rate decision verb in the statement: +1 hike / 0 hold / −1 cut. Mirrors the decision itself."),
        ("lexicon", "net hawkish−dovish stance words (transparent word-count). Goes silent on 2024+ statements, which drop stance adjectives."),
        ("RoBERTa", "a machine-learning per-sentence stance model, averaged. Populated on the heavy inference run; gapped until then."),
        ("themes", "how often the statement mentions each topic (inflation, employment, growth, balance sheet, financial conditions), per 1,000 words. Presence, not sentiment."),
        ("change magnitude", "how much the wording changed vs the prior statement (0 = identical, 1 = rewritten)."),
        ("communication style", "statement length, Flesch readability, and hedging/uncertainty word density over time."),
    ]
    lis = "".join(f"<li><b>{_html.escape(t)}</b> — {_html.escape(d)}</li>" for t, d in items)
    return ("<p>This is a <b>descriptive tracker</b>: it reads what each FOMC statement "
            "<i>says</i> and <i>how it changed</i> — not a rate forecast.</p>"
            f"<ul class='glossary'>{lis}</ul>")


def _fmt(v) -> str:
    return "—" if v is None else f"{v:+.3f}"


def _deltas_table_html(deltas: dict) -> str:
    if not deltas:
        return '<p class="rl-empty">Need ≥2 statements to compute deltas.</p>'
    rows = []
    for col, name in _LEVELS:
        m = deltas[col]
        rows.append(f"<tr><td>{name}</td><td>{_fmt(m['prior'])}</td>"
                    f"<td>{_fmt(m['latest'])}</td><td><b>{_fmt(m['delta'])}</b></td></tr>")
    return (f"<p>Comparing <b>{deltas['date_latest']}</b> vs <b>{deltas['date_prior']}</b></p>"
            "<table class='deltas'><tr><th>measure</th><th>prior</th><th>latest</th>"
            "<th>Δ</th></tr>" + "".join(rows) + "</table>")


def _selector_html(dates: list[str]) -> str:
    """Dropdown of every meeting that has a prior (all dates except the first),
    newest first, latest pre-selected. Empty string when <2 meetings."""
    pairs = dates[1:]
    if not pairs:
        return ""
    opts = "".join(f'<option value="{d}"{" selected" if d == pairs[-1] else ""}>{d}</option>'
                   for d in reversed(pairs))
    return ('<p class="selector">Show statement: '
            f'<select id="meeting">{opts}</select> <span>vs its prior</span></p>')


def _toggle_js(redlines_url: str) -> str:
    """Client-side swap: fetch the payload once, replace deltas+redline innerHTML,
    and move a dotted vertical marker on each chart via Plotly.relayout."""
    ids = ["fig-heatmap", "fig-change", "fig-commstyle", "fig-levels", "fig-deltas"]
    return """
<script>
const CHART_IDS = %s;
let _payload = null;
async function _loadPayload(){
  if(!_payload){ _payload = await fetch("%s").then(r => r.json()); }
  return _payload;
}
function _mark(date){
  const shapes = [{type:"line", xref:"x", yref:"paper", x0:date, x1:date,
                   y0:0, y1:1, line:{color:"#888", dash:"dot", width:1}}];
  CHART_IDS.forEach(id => {
    const el = document.getElementById(id);
    if(el && window.Plotly){ Plotly.relayout(el, {shapes: shapes}); }
  });
}
async function _onSelect(date){
  const map = await _loadPayload();
  const entry = map[date];
  if(entry){
    document.getElementById("deltas-slot").innerHTML = entry.deltas_html;
    document.getElementById("redline-slot").innerHTML = entry.redline_html;
  }
  _mark(date);
}
document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("meeting");
  if(sel){
    sel.addEventListener("change", e => _onSelect(e.target.value));
    _mark(sel.value);
  }
});
</script>""" % (json.dumps(ids), redlines_url)


def build_redlines_payload(deltas_by_date: dict, segments_by_date: dict) -> dict:
    """For every date present in `segments_by_date`, pre-render both panels.
    `deltas_by_date` maps date string -> an all_pair_deltas() entry; a date absent
    from it renders the empty-state deltas table. Returns
    {date: {"deltas_html": ..., "redline_html": ...}} — the toggle payload served as
    site/redlines.json. Both renderers stay the single source of truth."""
    payload = {}
    for date, segments in segments_by_date.items():
        payload[date] = {
            "deltas_html": _deltas_table_html(deltas_by_date.get(date, {})),
            "redline_html": build_redline_html(segments),
        }
    return payload


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}
.banner{background:#EAF2FB;border:1px solid #B6D4F2;padding:10px 14px;border-radius:6px;font-size:14px}
h1{font-size:23px}h2{font-size:17px;margin-top:30px;border-bottom:1px solid #eee;padding-bottom:4px}
ul.glossary{font-size:14px;line-height:1.5}ul.glossary li{margin:3px 0}
table.deltas{border-collapse:collapse}table.deltas td,table.deltas th{border:1px solid #ddd;padding:4px 10px;text-align:right}
table.deltas td:first-child,table.deltas th:first-child{text-align:left}
.redline-flow{line-height:1.7;font-size:15px}
.redline-flow span{padding:1px 2px;border-radius:3px}
.rl-equal{color:#444}.rl-insert{background:#E6F4EA;color:#137333}
.rl-delete{background:#FCE8E6;color:#A50E0E;text-decoration:line-through}
.rl-empty{color:#888;font-style:italic}
.selector{font-size:14px;margin:6px 0}.selector select{font-size:14px;padding:2px 4px}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FOMC Statement Tracker</title><style>{css}</style></head><body>
<h1>FOMC Statement Tracker</h1>
<h2>How to read this tracker</h2>
<div class="banner">{glossary}</div>
<h2>Latest statement — what changed vs the prior</h2>
{selector}
<div id="deltas-slot">{deltas_table}</div>
<div id="redline-slot" class="redline">{redline}</div>
<h2>What the Fed is focused on</h2>{heatmap}
<h2>How much each statement changed</h2>{change_fig}
<h2>Communication style</h2>{commstyle}
<h2>Stance measures, in context</h2>{levels}{deltas_fig}
{toggle_js}
</body></html>"""


def render_site(history: pd.DataFrame, deltas: dict, segments: list[dict],
                out_path: Path, *, verdict_url: str = VERDICT_URL,
                redlines_url: str = "redlines.json") -> None:
    """Assemble the tracker and write it to `out_path`. When history has >=2 rows,
    a meeting selector is emitted; picking a date swaps the deltas table + redline
    (from `redlines_url`, fetched once) and moves a marker on each chart.
    Only the first figure inlines plotly.js; the rest reference it."""
    heatmap = build_theme_heatmap(history).to_html(full_html=False, include_plotlyjs="inline", div_id="fig-heatmap")
    change_fig = build_change_magnitude_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-change")
    commstyle = build_commstyle_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-commstyle")
    levels = build_levels_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-levels")
    deltas_fig = build_deltas_figure(history).to_html(full_html=False, include_plotlyjs=False, div_id="fig-deltas")
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in history["date"]]
    page = _PAGE.format(css=_CSS, glossary=glossary_html(), selector=_selector_html(dates),
                        deltas_table=_deltas_table_html(deltas),
                        redline=build_redline_html(segments), heatmap=heatmap, change_fig=change_fig,
                        commstyle=commstyle, levels=levels, deltas_fig=deltas_fig,
                        toggle_js=_toggle_js(redlines_url))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
