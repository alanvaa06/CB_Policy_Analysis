# tests/test_monitor_main.py
import json
import pandas as pd
import pytest

pytest.importorskip("plotly")

from cbp.config import Config
from cbp.monitor import __main__ as m
from cbp.monitor.history import load_history


# fake statement HTML keyed by the URL's date stamp; returns hawkish/dovish bodies
_HTML = {
    "20240131": "<html><body><p>The Committee decided to raise the target range. "
                "Policy is restrictive.</p></body></html>",
    "20240320": "<html><body><p>The Committee decided to lower the target range. "
                "Policy is accommodative.</p></body></html>",
}


def _fake_get_html(url):
    for ymd, body in _HTML.items():
        if ymd in url:
            return body
    return None


def _fake_roberta(texts):
    return [{"label": "LABEL_2"} for _ in texts]  # neutral 0.0


def _cfg(tmp_path) -> Config:
    return Config(
        history_path=tmp_path / "tone_history.csv",
        calendar_path=tmp_path / "cal.csv",
        redline_path=tmp_path / "latest_redline.json",
        statements_dir=tmp_path / "statements",
        site_out=tmp_path / "site" / "index.html",
        lexicon_dir=__import__("pathlib").Path("data/lexicons"),
    )


def test_run_monitor_end_to_end_builds_history_redline_and_site(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.calendar_path.write_text("date\n2024-01-31\n2024-03-20\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)

    hist = load_history(cfg.history_path)
    assert list(hist["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-03-20"]
    assert hist.loc[hist["date"] == "2024-03-20", "action"].iloc[0] == -1.0

    redline = json.loads(cfg.redline_path.read_text())
    assert redline["date_latest"] == "2024-03-20"
    assert any(s["op"] in {"replace", "insert", "delete"} for s in redline["segments"])

    assert cfg.site_out.exists()
    assert "descriptive monitor" in cfg.site_out.read_text(encoding="utf-8").lower()


def test_rebuild_only_does_not_fetch(tmp_path):
    cfg = _cfg(tmp_path)
    # seed a history + redline; NO calendar, NO get_html -> must not touch the network
    cfg.calendar_path.write_text("date\n2024-01-31\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)
    cfg.site_out.unlink()
    # rebuild with a getter that would explode if called
    def _boom(url):
        raise AssertionError("rebuild-only must not fetch")
    m.run_monitor(cfg, rebuild_only=True, get_html=_boom)
    assert cfg.site_out.exists()
