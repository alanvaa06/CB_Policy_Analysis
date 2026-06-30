# src/cbp/models/lexicon_scorer.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic word tokens. Drops digits and
    punctuation. Deterministic and dependency-free."""
    return _WORD.findall(text.lower())


def _count_side(tokens: list[str], stems: frozenset[str]) -> int:
    return sum(1 for t in tokens if any(t.startswith(s) for s in stems))


def score_statement_lexicon(text: str, hawk: frozenset[str], dove: frozenset[str]) -> float:
    """Document-level net tone = (n_hawk - n_dove) / (n_hawk + n_dove) over all
    tokens in `text`. Returns 0.0 when neither side fires (a valid neutral
    measurement, not NaN)."""
    tokens = tokenize(text)
    h = _count_side(tokens, hawk)
    d = _count_side(tokens, dove)
    total = h + d
    return 0.0 if total == 0 else (h - d) / total


def score_statements_lexicon(
    statements: pd.DataFrame, hawk: frozenset[str], dove: frozenset[str]
) -> pd.DataFrame:
    """Score each statement's document-level net tone. Mirrors the output of
    `models.stance_scorer.score_statements`: columns [date, stance], one row per
    statement. Unlike the RoBERTa path, empty/keyword-less text is kept as a
    valid 0.0 (neutral) reading and logged, not skipped."""
    rows = []
    n_zero = 0
    for _, r in statements.iterrows():
        stance = score_statement_lexicon(r["text"], hawk, dove)
        if stance == 0.0:
            n_zero += 1
        rows.append({"date": r["date"], "stance": stance})
    if n_zero:
        logger.info("lexicon: %d/%d statements scored neutral 0.0 (no directional words)",
                    n_zero, len(statements))
    return pd.DataFrame(rows, columns=["date", "stance"])


def load_lexicon(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Load the hawkish/dovish stem lists from a JSON file.

    Returns (hawk, dove) as lowercased frozensets. Raises ValueError naming the
    path if the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hawk = frozenset(w.lower() for w in data["hawk"])
        dove = frozenset(w.lower() for w in data["dove"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"could not load lexicon from {path}: {e}") from e
    return hawk, dove
