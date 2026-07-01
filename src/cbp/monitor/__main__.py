# src/cbp/monitor/__main__.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from cbp.config import Config
from cbp.data.fomc_statements import fetch_statements
from cbp.models.stance_scorer import StanceClassifier
from cbp.monitor.calendar import load_calendar, pending_dates
from cbp.monitor.contrast import all_pair_deltas, redline, tone_deltas
from cbp.monitor.history import load_history, save_history, upsert_history
from cbp.monitor.score import score_all_measures
from cbp.monitor.site import VERDICT_URL, build_redlines_payload, render_site

logger = logging.getLogger(__name__)


def build_classifier(cfg: Config, use_roberta: bool) -> StanceClassifier | None:
    """Lazily build the real RoBERTa pipeline (needs [infer]); None when disabled."""
    if not use_roberta:
        return None
    from cbp.models.stance_scorer import load_fomc_roberta
    return load_fomc_roberta(cfg.roberta_model_id)


def _write_latest_redline(cfg: Config, history: pd.DataFrame) -> None:
    """Persist the redline of the two most recent statements so CI (which lacks the
    HTML cache) can render the panel. Reads both texts from the local cache."""
    if len(history) < 2:
        return
    last2 = [pd.Timestamp(d).date() for d in history["date"].iloc[-2:]]
    texts = fetch_statements(last2, cfg.statements_dir)  # cache hit; no network locally
    tmap = {pd.Timestamp(r.date).date(): r.text for r in texts.itertuples()}
    if last2[0] in tmap and last2[1] in tmap:
        payload = {"date_prior": str(last2[0]), "date_latest": str(last2[1]),
                   "segments": redline(tmap[last2[0]], tmap[last2[1]])}
        Path(cfg.redline_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.redline_path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _write_all_redlines(cfg: Config, history: pd.DataFrame) -> None:
    """Precompute the toggle payload for every consecutive pair and write it to
    cfg.redlines_path ({date: {deltas_html, redline_html}}). Reads all statement
    texts from the local cache; full-run only (CI lacks the raw texts)."""
    if len(history) < 2:
        return
    dates = [pd.Timestamp(d).date() for d in history["date"]]
    texts = fetch_statements(dates, cfg.statements_dir)   # cache hit; no network locally
    tmap = {pd.Timestamp(r.date).date(): r.text for r in texts.itertuples()}
    segments_by_date = {}
    for i in range(1, len(dates)):
        prev_d, curr_d = dates[i - 1], dates[i]
        if prev_d in tmap and curr_d in tmap:
            key = curr_d.strftime("%Y-%m-%d")
            segments_by_date[key] = redline(tmap[prev_d], tmap[curr_d])
    if len(segments_by_date) < len(dates) - 1:
        logger.warning("redlines: %d/%d pairs written; some cache texts missing",
                       len(segments_by_date), len(dates) - 1)
    payload = build_redlines_payload(all_pair_deltas(history), segments_by_date)
    Path(cfg.redlines_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.redlines_path).write_text(json.dumps(payload), encoding="utf-8")  # compact: large fetched artifact


def _load_segments(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("segments", [])


def run_monitor(
    cfg: Config,
    *,
    dates: list[dt.date] | None = None,
    use_roberta: bool = True,
    rebuild_only: bool = False,
    get_html=None,
    roberta: StanceClassifier | None = None,
) -> None:
    """Discover pending statements, score + upsert them, then render the dashboard.

    `get_html`/`roberta` are injection seams for tests; production passes neither and
    the real HTTP fetch + RoBERTa pipeline are used. `rebuild_only` skips all fetching
    and scoring and just re-renders from the committed CSV + redline JSON (the CI path)."""
    history = load_history(cfg.history_path)

    if not rebuild_only:
        todo = dates if dates is not None else pending_dates(load_calendar(cfg.calendar_path), history)
        if todo:
            kw = {} if get_html is None else {"get_html": get_html}
            statements = fetch_statements(todo, cfg.statements_dir, **kw)
            if not statements.empty:
                clf = roberta if roberta is not None else build_classifier(cfg, use_roberta)
                prior_text = None
                if len(history):
                    last_date = pd.Timestamp(history["date"].iloc[-1]).date()
                    prior = fetch_statements([last_date], cfg.statements_dir, **kw)
                    if not prior.empty:
                        prior_text = prior.iloc[0]["text"]
                scored = score_all_measures(statements, lexicon_dir=cfg.lexicon_dir,
                                            themes_path=cfg.themes_path, roberta=clf,
                                            prior_text=prior_text)
                history = upsert_history(history, scored)
                save_history(history, cfg.history_path)
                _write_latest_redline(cfg, history)
            else:
                logger.warning("no statements fetched for %d pending date(s)", len(todo))
        _write_all_redlines(cfg, history)   # every non-rebuild run keeps redlines.json in sync

    render_site(history, tone_deltas(history), _load_segments(cfg.redline_path),
                cfg.site_out, verdict_url=VERDICT_URL)
    logger.info("dashboard written to %s (%d statements)", cfg.site_out, len(history))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m cbp.monitor",
                                 description="FOMC statement monitor -> static dashboard")
    ap.add_argument("--date", help="process a single statement date YYYY-MM-DD (else all pending)")
    ap.add_argument("--no-roberta", action="store_true", help="torch-free run; roberta_stance=NaN")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="re-render HTML from the committed CSV + redline JSON (no fetch, no torch)")
    args = ap.parse_args()
    dates = [dt.date.fromisoformat(args.date)] if args.date else None
    run_monitor(Config(), dates=dates, use_roberta=not args.no_roberta,
                rebuild_only=args.rebuild_only)


if __name__ == "__main__":
    main()
