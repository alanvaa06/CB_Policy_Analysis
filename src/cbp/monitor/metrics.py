# src/cbp/monitor/metrics.py
from __future__ import annotations

import json
from pathlib import Path


def load_themes(path: Path) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Load the theme stem-lists + the uncertainty stem-list from JSON.

    Returns (themes, uncertainty) with all stems lowercased. Raises ValueError
    naming the path if the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        themes = {k: frozenset(s.lower() for s in v) for k, v in data["themes"].items()}
        uncertainty = frozenset(s.lower() for s in data["uncertainty"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"could not load themes from {path}: {e}") from e
    return themes, uncertainty
