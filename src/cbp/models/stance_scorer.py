# src/cbp/models/stance_scorer.py
from __future__ import annotations

import logging
import re
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, float] = {"LABEL_0": -1.0, "LABEL_1": 1.0, "LABEL_2": 0.0}  # Dovish/Hawkish/Neutral

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


class StanceClassifier(Protocol):
    def __call__(self, texts: list[str]) -> list[dict]: ...


def score_statements(statements: pd.DataFrame, classifier: StanceClassifier) -> pd.DataFrame:
    """Score each statement's stance = mean of mapped per-sentence labels.

    Splits each statement into sentences, classifies each with the injected
    `classifier` (HF-pipeline shape: [{"label": "LABEL_x", ...}, ...]), maps
    labels via LABEL_MAP, and averages. Statements with no sentences are skipped
    and logged. Returns columns [date, stance].
    """
    rows = []
    for _, r in statements.iterrows():
        sentences = split_sentences(r["text"])
        if not sentences:
            logger.warning("No sentences for statement %s; skipping", r["date"])
            continue
        preds = classifier(sentences)
        mapped = [LABEL_MAP[p["label"]] for p in preds]
        rows.append({"date": r["date"], "stance": float(np.mean(mapped))})
    return pd.DataFrame(rows, columns=["date", "stance"])
