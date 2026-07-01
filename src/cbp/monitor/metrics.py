# src/cbp/monitor/metrics.py
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from cbp.models.lexicon_scorer import tokenize
from cbp.models.stance_scorer import split_sentences

# Boilerplate markers (conservative; DOTALL so a match runs to end of text).
_VOTING = re.compile(r"\bVoting (?:for|against)\b.*", re.IGNORECASE | re.DOTALL)
_MEDIA = re.compile(r"\bFor media inquiries\b.*", re.IGNORECASE | re.DOTALL)
_IMPL = re.compile(r"\bImplementation Note issued\b.*", re.IGNORECASE | re.DOTALL)
_RELEASE_HDR = re.compile(r"^.*?\b(?:EDT|EST)\b\s*(?:Share\s+)?", re.IGNORECASE | re.DOTALL)
_VOWELS = re.compile(r"[aeiouy]+")

# UTF-8 bytes mis-decoded as Latin-1 (mojibake) surface as a lead char U+00C2–U+00F4
# followed by continuation chars U+0080–U+00BF, e.g. the en-dash E2 80 93 -> "â\x80\x93".
# We repair each such run independently (per-match), so a genuine non-Latin-1 character
# elsewhere in the text cannot defeat the whole repair.
_MOJIBAKE_SEQ = re.compile("[Â-ô][-¿]+")


def _fix_mojibake(text: str) -> str:
    """Repair UTF-8-decoded-as-Latin-1 mojibake runs by re-encoding each matched run to
    Latin-1 bytes and decoding as UTF-8. Non-mojibake text is left untouched; a run that
    does not round-trip cleanly is kept as-is."""
    def _repl(m: "re.Match[str]") -> str:
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)
    return _MOJIBAKE_SEQ.sub(_repl, text)


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


def clean_statement(text: str) -> str:
    """Strip clearly-identified boilerplate (release header, voting roster, media line,
    implementation note) and repair mojibake so metrics + the redline read the substance
    only. Conservative: each cut is anchored on an explicit marker; historical statements
    without these pass through. Never returns empty — falls back to the stripped raw text."""
    text = _fix_mojibake(text)
    s = _VOTING.sub("", text)
    s = _MEDIA.sub("", s)
    s = _IMPL.sub("", s)
    stripped_head = _RELEASE_HDR.sub("", s, count=1)
    s = stripped_head if stripped_head.strip() else s
    s = s.strip()
    return s if s else text.strip()


def word_count(text: str) -> int:
    """Number of alphabetic word tokens (reuses the lexicon tokenizer)."""
    return len(tokenize(text))


def _syllables(word: str) -> int:
    return max(1, len(_VOWELS.findall(word)))


def flesch(text: str) -> float:
    """Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words).
    Syllables via a vowel-group heuristic (approximate; read the trend, not the absolute).
    Returns 0.0 when there are no words or sentences."""
    words = tokenize(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    syll = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syll / len(words))


def _count_prefix(tokens: list[str], stems: frozenset[str]) -> int:
    return sum(1 for t in tokens if any(t.startswith(s) for s in stems))


def count_themes(text: str, themes: dict[str, frozenset[str]]) -> dict[str, int]:
    """Raw token-prefix hit count per theme."""
    tokens = tokenize(text)
    return {name: _count_prefix(tokens, stems) for name, stems in themes.items()}


def uncertainty_count(text: str, terms: frozenset[str]) -> int:
    """Raw token-prefix hit count for the uncertainty stem-list."""
    return _count_prefix(tokenize(text), terms)


def change_magnitude(prev_text: str, curr_text: str) -> float:
    """1 - difflib word-ratio between two statements, in [0,1]. 0 = identical wording,
    1 = fully rewritten. Coarse edit size; pair with the redline for *what* changed."""
    a, b = tokenize(prev_text), tokenize(curr_text)
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
