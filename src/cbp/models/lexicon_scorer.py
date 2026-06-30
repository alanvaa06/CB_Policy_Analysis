# src/cbp/models/lexicon_scorer.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

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
