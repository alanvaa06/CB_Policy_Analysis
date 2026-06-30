# src/cbp/monitor/score.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
from cbp.models.stance_scorer import StanceClassifier, score_statements, split_sentences
from cbp.monitor.history import HISTORY_COLUMNS


def score_all_measures(
    statements: pd.DataFrame,
    *,
    lexicon_dir: Path,
    roberta: StanceClassifier | None = None,
) -> pd.DataFrame:
    """Score each statement on all three descriptive measures and return one row per
    statement with exactly HISTORY_COLUMNS.

    - action       : action_tone.json lexicon (+1 raise / -1 lower / 0 hold)
    - lexicon_tone  : hawk_dove.json lexicon net stance
    - roberta_stance: per-sentence-mean stance via the injected `roberta` classifier;
                      NaN (whole column) when `roberta is None`. Statements RoBERTa skips
                      (no sentences) become NaN via the left-merge.
    - n_sentences   : split_sentences count (independent of RoBERTa availability)
    """
    lexicon_dir = Path(lexicon_dir)
    hawk_a, dove_a = load_lexicon(lexicon_dir / "action_tone.json")
    hawk_l, dove_l = load_lexicon(lexicon_dir / "hawk_dove.json")

    act = score_statements_lexicon(statements, hawk_a, dove_a).rename(columns={"stance": "action"})
    lex = score_statements_lexicon(statements, hawk_l, dove_l).rename(columns={"stance": "lexicon_tone"})
    nsent = pd.DataFrame({
        "date": statements["date"].to_numpy(),
        "n_sentences": [len(split_sentences(t)) for t in statements["text"]],
    })

    out = act.merge(lex, on="date").merge(nsent, on="date")
    if roberta is not None:
        rob = score_statements(statements, roberta).rename(columns={"stance": "roberta_stance"})
        out = out.merge(rob, on="date", how="left")
    else:
        out["roberta_stance"] = float("nan")
    return out.reindex(columns=HISTORY_COLUMNS)
