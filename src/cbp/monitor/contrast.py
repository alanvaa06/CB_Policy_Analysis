# src/cbp/monitor/contrast.py
from __future__ import annotations

import difflib
import re

import pandas as pd

from cbp.monitor.metrics import clean_statement

_WORDS = re.compile(r"\w+|[^\w\s]")

MEASURES = ["action", "lexicon_tone", "roberta_stance"]


def _num(v):
    return None if pd.isna(v) else float(v)


def tone_deltas(history: pd.DataFrame) -> dict:
    """Latest-vs-prior change per measure, from the last two history rows.

    Returns {} when fewer than two rows. Otherwise:
      {"date_prior": "YYYY-MM-DD", "date_latest": "YYYY-MM-DD",
       <measure>: {"prior": float|None, "latest": float|None, "delta": float|None}}
    `delta` is None when either side is NaN (e.g. a --no-roberta gap)."""
    if len(history) < 2:
        return {}
    prior, latest = history.iloc[-2], history.iloc[-1]
    out = {
        "date_prior": pd.Timestamp(prior["date"]).strftime("%Y-%m-%d"),
        "date_latest": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
    }
    for m in MEASURES:
        p, l = _num(prior[m]), _num(latest[m])
        out[m] = {"prior": p, "latest": l,
                  "delta": (None if p is None or l is None else l - p)}
    return out


def redline(prev_text: str, curr_text: str) -> list[dict]:
    """Word-level track-changes diff of two statements, over boilerplate-stripped text.

    Cleans both with clean_statement, tokenizes to whitespace-separated word runs, runs
    difflib, and emits ordered segments {op, prev, curr} with
    op in {equal, insert, delete, replace}. Reads as one paragraph with only the changed
    words highlighted (vs the v1 sentence-block walls). Textual, not semantic."""
    a = _WORDS.findall(clean_statement(prev_text))
    b = _WORDS.findall(clean_statement(curr_text))
    segments: list[dict] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        prev = " ".join(a[i1:i2])
        curr = " ".join(b[j1:j2])
        if op == "insert":
            prev = ""
        elif op == "delete":
            curr = ""
        segments.append({"op": op, "prev": prev, "curr": curr})
    return segments
