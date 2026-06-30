# src/cbp/monitor/site.py
from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

VERDICT_URL = "https://github.com/"  # replaced with the repo verdict link in Task 10
_LEVELS = [("action", "action"), ("lexicon_tone", "lexicon"), ("roberta_stance", "RoBERTa")]


def build_levels_figure(history: pd.DataFrame):
    """Tone levels through time: one line per measure, gaps left as gaps."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col], name=name,
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="FOMC statement tone — levels (1999→latest)",
                      yaxis_title="tone", template="plotly_white", height=380)
    return fig


def build_deltas_figure(history: pd.DataFrame):
    """Meeting-over-meeting Δ per measure (the delta-history)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col].diff(), name=f"Δ {name}",
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="FOMC statement tone — meeting-over-meeting change",
                      yaxis_title="Δ tone", template="plotly_white", height=380)
    return fig


def build_redline_html(segments: list[dict]) -> str:
    """Render redline segments as <p> blocks with rl-<op> CSS classes."""
    if not segments:
        return '<p class="rl-empty">Need ≥2 statements to show a redline.</p>'
    rows = []
    for s in segments:
        op = s["op"]
        if op == "equal":
            rows.append(f'<p class="rl-equal">{_html.escape(s["curr"])}</p>')
        elif op == "insert":
            rows.append(f'<p class="rl-insert">+ {_html.escape(s["curr"])}</p>')
        elif op == "delete":
            rows.append(f'<p class="rl-delete">− {_html.escape(s["prev"])}</p>')
        else:  # replace
            rows.append(f'<p class="rl-replace">− {_html.escape(s["prev"])}'
                        f'<br>+ {_html.escape(s["curr"])}</p>')
    return "\n".join(rows)


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


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}
.banner{background:#FFF3CD;border:1px solid #E0C97A;padding:10px 14px;border-radius:6px;font-size:14px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px}
table.deltas{border-collapse:collapse}table.deltas td,table.deltas th{border:1px solid #ddd;padding:4px 10px;text-align:right}
table.deltas td:first-child,table.deltas th:first-child{text-align:left}
.redline p{margin:4px 0;padding:4px 8px;border-radius:4px}
.rl-equal{color:#555}.rl-insert{background:#E6F4EA;color:#137333}
.rl-delete{background:#FCE8E6;color:#A50E0E;text-decoration:line-through}
.rl-replace{background:#FEF7E0;color:#8A6D00}.rl-empty{color:#888;font-style:italic}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FOMC Statement Monitor</title><style>{css}</style></head><body>
<h1>FOMC Statement Monitor</h1>
<div class="banner">{banner}</div>
<h2>Latest vs prior — tone deltas</h2>{deltas_table}
<h2>Latest vs prior — statement redline</h2><div class="redline">{redline}</div>
<h2>Tone levels through time</h2>{levels}
<h2>Tone change through time</h2>{deltas_fig}
</body></html>"""


def render_site(history: pd.DataFrame, deltas: dict, segments: list[dict],
                out_path: Path, *, verdict_url: str = VERDICT_URL) -> None:
    """Assemble the self-contained dashboard and write it to `out_path`."""
    levels = build_levels_figure(history).to_html(full_html=False, include_plotlyjs="inline")
    deltas_fig = build_deltas_figure(history).to_html(full_html=False, include_plotlyjs=False)
    banner = ("This is a <b>descriptive monitor, not a predictive signal</b>. FOMC statement "
              "tone adds no out-of-sample value beyond the policy surprise "
              f'(<a href="{verdict_url}">Phase 1/2a verdicts</a>).')
    page = _PAGE.format(css=_CSS, banner=banner, deltas_table=_deltas_table_html(deltas),
                        redline=build_redline_html(segments), levels=levels, deltas_fig=deltas_fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
