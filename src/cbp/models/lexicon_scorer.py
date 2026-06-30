# src/cbp/models/lexicon_scorer.py
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
