# src/cbp/monitor/score.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
from cbp.models.stance_scorer import StanceClassifier, score_statements, split_sentences
from cbp.monitor.history import HISTORY_COLUMNS
from cbp.monitor.metrics import (
    change_magnitude, clean_statement, count_themes, flesch, load_themes,
    uncertainty_count, word_count,
)


def _metric_rows(statements: pd.DataFrame, themes_path: Path, prior_text: str | None) -> pd.DataFrame:
    """Per-statement text metrics + change_magnitude vs the previous statement.

    `statements` must be date-sorted. change_magnitude for the first row is measured
    against `prior_text` (the statement before this batch) or NaN if none."""
    themes, uncertainty = load_themes(themes_path)
    cleans = [clean_statement(t) for t in statements["text"]]
    prev_clean = clean_statement(prior_text) if prior_text else None
    rows = []
    for i, clean in enumerate(cleans):
        wc = word_count(clean)
        per1k = (1000.0 / wc) if wc else 0.0
        theme_hits = count_themes(clean, themes)
        if i > 0:
            cm = change_magnitude(cleans[i - 1], clean)
        elif prev_clean is not None:
            cm = change_magnitude(prev_clean, clean)
        else:
            cm = float("nan")
        row = {
            "date": statements["date"].iloc[i],
            "word_count": wc,
            "flesch": flesch(clean),
            "uncertainty_per1k": uncertainty_count(clean, uncertainty) * per1k,
            "change_magnitude": cm,
        }
        for name, hits in theme_hits.items():
            row[f"theme_{name}"] = hits * per1k
        rows.append(row)
    return pd.DataFrame(rows)


def score_all_measures(
    statements: pd.DataFrame,
    *,
    lexicon_dir: Path,
    themes_path: Path,
    roberta: StanceClassifier | None = None,
    prior_text: str | None = None,
) -> pd.DataFrame:
    """Score each statement on the three stance measures AND the descriptive text
    metrics (length, readability, theme intensity, uncertainty, change-magnitude),
    returning one row per statement with exactly HISTORY_COLUMNS.

    - action / lexicon_tone / roberta_stance / n_sentences: as in v1.
    - word_count, flesch, uncertainty_per1k, theme_*: per-statement on clean_statement(text).
    - change_magnitude: word edit-distance vs the prior statement (NaN for the first ever).
    `prior_text` lets an incremental run measure change_magnitude against the statement
    already in history; `statements` is sorted by date before scoring.
    """
    statements = statements.sort_values("date").reset_index(drop=True)
    lexicon_dir = Path(lexicon_dir)
    hawk_a, dove_a = load_lexicon(lexicon_dir / "action_tone.json")
    hawk_l, dove_l = load_lexicon(lexicon_dir / "hawk_dove.json")

    act = score_statements_lexicon(statements, hawk_a, dove_a).rename(columns={"stance": "action"})
    lex = score_statements_lexicon(statements, hawk_l, dove_l).rename(columns={"stance": "lexicon_tone"})
    nsent = pd.DataFrame({
        "date": statements["date"].to_numpy(),
        "n_sentences": [len(split_sentences(t)) for t in statements["text"]],
    })
    metrics = _metric_rows(statements, Path(themes_path), prior_text)

    out = act.merge(lex, on="date").merge(nsent, on="date").merge(metrics, on="date")
    if roberta is not None:
        rob = score_statements(statements, roberta).rename(columns={"stance": "roberta_stance"})
        out = out.merge(rob, on="date", how="left")
    else:
        out["roberta_stance"] = float("nan")
    return out[HISTORY_COLUMNS]
