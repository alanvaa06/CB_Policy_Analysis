# src/cbp/monitor/contrast.py
from __future__ import annotations

import difflib
import re

import pandas as pd

from cbp.monitor.metrics import clean_statement

_WORDS = re.compile(r"\w+|[^\w\s]")
_TIGHT_BEFORE = set(".,;:%!?)]}")   # no space before these tokens
_TIGHT_AFTER = set("([{$")          # no space after these tokens

MEASURES = ["action", "lexicon_tone", "roberta_stance"]


def _smart_join(tokens: list[str]) -> str:
    """Join word/punctuation tokens back into natural prose: no space before closing
    punctuation, after opening punctuation, or around apostrophes (`'s`), hyphens, and
    slashes (so `3 - 1 / 2` → `3-1/2`, `Reserve ' s` → `Reserve's`)."""
    out: list[str] = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            continue
        prev = tokens[i - 1]
        tight = (
            t[:1] in _TIGHT_BEFORE
            or prev[-1:] in _TIGHT_AFTER
            or t in ("'", "-", "/")
            or prev in ("'", "-", "/")
        )
        out.append(t if tight else " " + t)
    return "".join(out)


def _num(v):
    return None if pd.isna(v) else float(v)


def _pair_deltas(prior: pd.Series, latest: pd.Series) -> dict:
    """Change per measure between two history rows (prior -> latest)."""
    out = {
        "date_prior": pd.Timestamp(prior["date"]).strftime("%Y-%m-%d"),
        "date_latest": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
    }
    for m in MEASURES:
        p, l = _num(prior[m]), _num(latest[m])
        out[m] = {"prior": p, "latest": l,
                  "delta": (None if p is None or l is None else l - p)}
    return out


def tone_deltas(history: pd.DataFrame) -> dict:
    """Latest-vs-prior change per measure, from the last two history rows.

    Returns {} when fewer than two rows. Otherwise:
      {"date_prior": "YYYY-MM-DD", "date_latest": "YYYY-MM-DD",
       <measure>: {"prior": float|None, "latest": float|None, "delta": float|None}}
    `delta` is None when either side is NaN (e.g. a --no-roberta gap)."""
    if len(history) < 2:
        return {}
    return _pair_deltas(history.iloc[-2], history.iloc[-1])


def all_pair_deltas(history: pd.DataFrame) -> dict:
    """Deltas for every consecutive pair, keyed by the latest date string.
    Returns {} when fewer than two rows. Assumes `history` is date-sorted
    ascending with unique dates (as load_history/upsert_history guarantee)."""
    if len(history) < 2:
        return {}
    out = {}
    for i in range(1, len(history)):
        d = _pair_deltas(history.iloc[i - 1], history.iloc[i])
        out[d["date_latest"]] = d
    return out


def redline(prev_text: str, curr_text: str) -> list[dict]:
    """Word-level track-changes diff of two statements, over boilerplate-stripped text.

    Cleans both with clean_statement, tokenizes to word/punctuation tokens (punctuation
    diffs separately for precise change spans), runs difflib, and emits ordered segments
    {op, prev, curr} with op in {equal, insert, delete, replace}. Segment text is
    re-joined with `_smart_join` so it reads as natural prose (no spaced-out punctuation),
    with only the changed words highlighted. Textual, not semantic."""
    a = _WORDS.findall(clean_statement(prev_text))
    b = _WORDS.findall(clean_statement(curr_text))
    segments: list[dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        prev = _smart_join(a[i1:i2])
        curr = _smart_join(b[j1:j2])
        if op == "insert":
            prev = ""
        elif op == "delete":
            curr = ""
        segments.append({"op": op, "prev": prev, "curr": curr})
    return segments
