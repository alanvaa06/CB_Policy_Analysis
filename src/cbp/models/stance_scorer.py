from __future__ import annotations
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences on terminal punctuation followed by whitespace.

    Deterministic and dependency-free. A crude splitter (abbreviations like "U.S."
    may over-split) — acceptable because stance is a mean over sentences; recorded
    as a caveat (PRD §11). Returns [] for empty/blank input.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]
