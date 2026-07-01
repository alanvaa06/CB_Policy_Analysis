# FOMC Statement Tracker (Dashboard v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the v1 monitor into a descriptive "FOMC Statement Tracker": a readable word-level redline, depth analytics (themes, change-magnitude, communication style) computed torch-free from text, and an on-page glossary — all additive to the committed `tone_history.csv` render contract.

**Architecture:** Add one pure module `src/cbp/monitor/metrics.py` + a committed `data/lexicons/themes.json`; extend `HISTORY_COLUMNS`, `score_all_measures` (now also emits the metric columns + change-magnitude vs prior), `contrast.redline` (now word-level over cleaned text), `__main__.run_monitor` (passes prior text), and `site.py` (6-panel layout incl. glossary + theme heatmap). The render path stays torch-free and reads only the committed CSV + `latest_redline.json`.

**Tech Stack:** Python 3.11+, pandas, stdlib `json`/`re`/`difflib`, pytest. Plotly (`[site]` extra) isolated to `site.py`. No new runtime dependency (Flesch is in-repo). torch (`[infer]`) only to populate `roberta_stance`, never for tests/CI.

**Spec:** `docs/prd/005-statement-tracker-v2.md`

**Reused v1 signatures (verified in repo):**
- `cbp.models.lexicon_scorer.tokenize(text) -> list[str]` (lowercase alphabetic tokens)
- `cbp.models.stance_scorer.split_sentences(text) -> list[str]`
- `cbp.monitor.history.HISTORY_COLUMNS` (currently the 5 base columns); `load_history` fills missing columns with NA (legacy-tolerant); `upsert_history`/`save_history` operate over `HISTORY_COLUMNS`.
- `cbp.monitor.score.score_all_measures(statements, *, lexicon_dir, roberta=None)` — extended here.
- `cbp.monitor.contrast.redline(prev, curr)` / `tone_deltas(history)` — redline rewritten here.

---

### Task 1: Theme lexicon + loader + config path

**Files:**
- Create: `data/lexicons/themes.json`
- Create: `src/cbp/monitor/metrics.py` (starts with `load_themes` only; more functions added in Task 2)
- Modify: `src/cbp/config.py` (add `themes_path`)
- Test: `tests/test_monitor_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor_metrics.py
import json
from pathlib import Path

import pytest

from cbp.monitor.metrics import load_themes


def test_load_themes_returns_lowercase_frozensets(tmp_path):
    p = tmp_path / "themes.json"
    p.write_text(json.dumps({
        "themes": {"inflation": ["Inflat", "PRICE"], "growth": ["growth"]},
        "uncertainty": ["Uncertain", "risk"],
    }))
    themes, unc = load_themes(p)
    assert themes["inflation"] == frozenset({"inflat", "price"})
    assert themes["growth"] == frozenset({"growth"})
    assert unc == frozenset({"uncertain", "risk"})


def test_load_themes_missing_raises_valueerror():
    with pytest.raises(ValueError, match="themes"):
        load_themes(Path("does/not/exist.json"))


def test_repo_themes_file_has_five_themes_and_uncertainty():
    themes, unc = load_themes(Path("data/lexicons/themes.json"))
    assert set(themes) == {"inflation", "employment", "growth",
                           "balance_sheet", "financial_conditions"}
    assert all(len(v) >= 3 for v in themes.values())
    assert len(unc) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cbp.monitor.metrics'`

- [ ] **Step 3: Create `data/lexicons/themes.json`**

```json
{
  "themes": {
    "inflation": ["inflat", "price", "pce", "deflat", "disinflat"],
    "employment": ["employ", "labor", "labour", "job", "unemploy", "payroll", "wage", "hiring"],
    "growth": ["growth", "activ", "spend", "demand", "output", "expansion", "gdp", "product", "invest", "consum"],
    "balance_sheet": ["securit", "holding", "reserve", "reinvest", "runoff", "mortgage", "treasur", "redempt"],
    "financial_conditions": ["financ", "credit", "bank", "lending", "market", "borrow"]
  },
  "uncertainty": ["uncertain", "risk", "depend", "could", "evolv", "carefully", "assess"],
  "sources": ["seed list, corpus-checkable against cached statements"],
  "notes": "Single-token stems matched by token.startswith (like hawk_dove.json). Intensity = hits per 1000 words; descriptive presence, not sentiment."
}
```

- [ ] **Step 4: Create `metrics.py` with `load_themes`**

```python
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
```

- [ ] **Step 5: Add `themes_path` to config**

```python
# src/cbp/config.py — add this field inside the Config dataclass, after lexicon_dir
    themes_path: Path = Path("data/lexicons/themes.json")
```

- [ ] **Step 6: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add data/lexicons/themes.json src/cbp/monitor/metrics.py src/cbp/config.py tests/test_monitor_metrics.py
git commit -m "feat(tracker): theme lexicon + load_themes + config themes_path"
```

---

### Task 2: Text metrics — clean, word_count, flesch, themes, uncertainty, change_magnitude

**Files:**
- Modify: `src/cbp/monitor/metrics.py`
- Test: `tests/test_monitor_metrics.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_monitor_metrics.py — APPEND
import math

from cbp.monitor.metrics import (
    clean_statement, word_count, flesch, count_themes, uncertainty_count, change_magnitude,
)

_MODERN = ("April 29, 2026 For release at 2:00 p.m. EDT Share Recent indicators suggest "
           "that economic activity has been expanding at a solid pace. Inflation is elevated. "
           "Voting for the monetary policy action were Jerome H. Powell and others. "
           "For media inquiries, please email someone or call 202-452-2955.")


def test_clean_statement_strips_header_voting_and_media():
    c = clean_statement(_MODERN)
    assert c.startswith("Recent indicators")          # release header + "Share" gone
    assert "Voting for" not in c                        # voting roster gone
    assert "media inquiries" not in c                   # media line gone
    assert "Inflation is elevated." in c                # substance kept


def test_clean_statement_passthrough_when_no_boilerplate():
    raw = "The Committee decided to maintain the target range."
    assert clean_statement(raw) == raw


def test_clean_statement_never_empty():
    assert clean_statement("   ") == ""  # strip of blank -> "" via fallback to stripped raw


def test_word_count_counts_alphabetic_tokens():
    assert word_count("Rates at 5.25% will hold.") == 4  # rates, at, will, hold


def test_flesch_is_higher_for_simpler_text():
    simple = "The cat sat. The dog ran."
    complex_ = ("Notwithstanding heterogeneous macroprudential considerations, the Committee "
                "reaffirmed its accommodative configuration.")
    assert flesch(simple) > flesch(complex_)


def test_flesch_empty_is_zero():
    assert flesch("   ") == 0.0


def test_count_themes_prefix_matches():
    themes = {"inflation": frozenset({"inflat", "price"}), "growth": frozenset({"growth"})}
    counts = count_themes("Inflation and prices rose; growth slowed and inflationary risk", themes)
    assert counts["inflation"] == 3   # Inflation, prices, inflationary
    assert counts["growth"] == 1


def test_uncertainty_count():
    assert uncertainty_count("risks remain and the outcome is uncertain",
                             frozenset({"risk", "uncertain"})) == 2


def test_change_magnitude_bounds():
    assert change_magnitude("a b c d", "a b c d") == 0.0           # identical
    assert change_magnitude("a b c", "x y z") == 1.0               # disjoint
    mid = change_magnitude("the committee will hold rates steady",
                           "the committee will raise rates sharply")
    assert 0.0 < mid < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_metrics.py -k "clean or word_count or flesch or count_themes or uncertainty or change_magnitude" -v`
Expected: FAIL with `ImportError: cannot import name 'clean_statement'`

- [ ] **Step 3: Add the metric functions to `metrics.py`**

```python
# src/cbp/monitor/metrics.py — add imports at top (keep the existing json/Path imports)
import difflib
import re

from cbp.models.lexicon_scorer import tokenize
from cbp.models.stance_scorer import split_sentences

# Boilerplate markers (conservative; DOTALL so a match runs to end of text).
_VOTING = re.compile(r"\bVoting (?:for|against)\b.*", re.IGNORECASE | re.DOTALL)
_MEDIA = re.compile(r"\bFor media inquiries\b.*", re.IGNORECASE | re.DOTALL)
_IMPL = re.compile(r"\bImplementation Note issued\b.*", re.IGNORECASE | re.DOTALL)
_RELEASE_HDR = re.compile(r"^.*?\b(?:EDT|EST)\b\s*(?:Share\s+)?", re.IGNORECASE | re.DOTALL)
_VOWELS = re.compile(r"[aeiouy]+")
_WORDS = re.compile(r"\S+")


def clean_statement(text: str) -> str:
    """Strip clearly-identified boilerplate (release header, voting roster, media line,
    implementation note) so metrics + the redline read the substance only. Conservative:
    each cut is anchored on an explicit marker; historical statements without these pass
    through. Never returns empty — falls back to the stripped raw text."""
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_metrics.py -v`
Expected: PASS (all metric tests)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/metrics.py tests/test_monitor_metrics.py
git commit -m "feat(tracker): text metrics — clean/flesch/themes/uncertainty/change_magnitude"
```

---

### Task 3: Extend the tone-history schema

**Files:**
- Modify: `src/cbp/monitor/history.py:8`
- Test: `tests/test_monitor_history.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_monitor_history.py — APPEND
from cbp.monitor.history import METRIC_COLUMNS, THEME_COLUMNS


def test_history_columns_include_metric_and_theme_columns():
    assert THEME_COLUMNS == ["theme_inflation", "theme_employment", "theme_growth",
                             "theme_balance_sheet", "theme_financial_conditions"]
    for c in ["word_count", "flesch", "uncertainty_per1k", "change_magnitude", *THEME_COLUMNS]:
        assert c in METRIC_COLUMNS
        assert c in HISTORY_COLUMNS


def test_extended_row_roundtrips(tmp_path):
    p = tmp_path / "hist.csv"
    row = {c: 0.0 for c in HISTORY_COLUMNS}
    row["date"] = pd.Timestamp("2024-01-31")
    row["theme_inflation"] = 12.5
    row["change_magnitude"] = 0.4
    save_history(pd.DataFrame([row]), p)
    back = load_history(p)
    assert back.loc[0, "theme_inflation"] == 12.5
    assert back.loc[0, "change_magnitude"] == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_history.py -k "metric or extended" -v`
Expected: FAIL with `ImportError: cannot import name 'METRIC_COLUMNS'`

- [ ] **Step 3: Extend `HISTORY_COLUMNS`**

Replace the single `HISTORY_COLUMNS = [...]` line in `src/cbp/monitor/history.py` with:

```python
THEME_COLUMNS = ["theme_inflation", "theme_employment", "theme_growth",
                 "theme_balance_sheet", "theme_financial_conditions"]
METRIC_COLUMNS = ["word_count", "flesch", "uncertainty_per1k", "change_magnitude", *THEME_COLUMNS]
HISTORY_COLUMNS = ["date", "action", "lexicon_tone", "roberta_stance", "n_sentences", *METRIC_COLUMNS]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_history.py -v`
Expected: PASS (existing history tests + 2 new). `load_history` already fills missing columns, so legacy CSVs still load.

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/history.py tests/test_monitor_history.py
git commit -m "feat(tracker): extend tone_history schema with metric + theme columns"
```

---

### Task 4: Emit metrics from `score_all_measures`

**Files:**
- Modify: `src/cbp/monitor/score.py`
- Test: `tests/test_monitor_score.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_monitor_score.py — APPEND
from cbp.monitor.history import HISTORY_COLUMNS, THEME_COLUMNS

THEMES = "data/lexicons/themes.json"


def _two_statements():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "text": ["Inflation is elevated and prices rose. The Committee will raise rates.",
                 "Inflation is elevated and prices rose. The Committee will raise rates again."],
    })


def test_score_all_measures_emits_metric_columns():
    out = score_all_measures(_two_statements(), lexicon_dir="data/lexicons",
                             themes_path=THEMES, roberta=None)
    assert list(out.columns) == HISTORY_COLUMNS
    # metrics populated (not NaN) for both rows
    for c in ["word_count", "flesch", "uncertainty_per1k", *THEME_COLUMNS]:
        assert out[c].notna().all()
    assert (out["theme_inflation"] > 0).all()          # "inflation"/"prices" fire
    # change_magnitude: first row has no prior -> NaN; second is a small edit (>0, <1)
    assert math.isnan(out.loc[0, "change_magnitude"])
    assert 0.0 < out.loc[1, "change_magnitude"] < 1.0


def test_score_all_measures_change_magnitude_uses_prior_text():
    one = _two_statements().iloc[[0]]
    out = score_all_measures(one, lexicon_dir="data/lexicons", themes_path=THEMES,
                             roberta=None, prior_text="A completely different earlier statement.")
    assert 0.0 < out.loc[0, "change_magnitude"] <= 1.0   # measured vs prior_text, not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_score.py -k "metric or prior_text" -v`
Expected: FAIL with `TypeError: score_all_measures() got an unexpected keyword argument 'themes_path'`

- [ ] **Step 3: Rewrite `score.py`**

```python
# src/cbp/monitor/score.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
from cbp.models.stance_scorer import StanceClassifier, score_statements, split_sentences
from cbp.monitor.history import HISTORY_COLUMNS
from cbp.monitor.metrics import (
    change_magnitude, clean_statement, count_themes, flesch, load_themes,
    uncertainty_count, word_count,
)


def _metric_rows(statements: pd.DataFrame, themes_path: Path, prior_text: str | None) -> pd.DataFrame:
    """Per-statement text metrics + change_magnitude vs the previous statement.

    `statements` must be date-sorted. change_magnitude for the first row is measured
    against `prior_text` (the statement before this batch) or NaN if none."""
    themes, uncertainty = load_themes(themes_path)
    cleans = [clean_statement(t) for t in statements["text"]]
    prev_clean = clean_statement(prior_text) if prior_text else None
    rows = []
    for i, clean in enumerate(cleans):
        wc = word_count(clean)
        per1k = (1000.0 / wc) if wc else 0.0
        theme_hits = count_themes(clean, themes)
        if i > 0:
            cm = change_magnitude(cleans[i - 1], clean)
        elif prev_clean is not None:
            cm = change_magnitude(prev_clean, clean)
        else:
            cm = float("nan")
        row = {
            "date": statements["date"].iloc[i],
            "word_count": wc,
            "flesch": flesch(clean),
            "uncertainty_per1k": uncertainty_count(clean, uncertainty) * per1k,
            "change_magnitude": cm,
        }
        for name, hits in theme_hits.items():
            row[f"theme_{name}"] = hits * per1k
        rows.append(row)
    return pd.DataFrame(rows)


def score_all_measures(
    statements: pd.DataFrame,
    *,
    lexicon_dir: Path,
    themes_path: Path,
    roberta: StanceClassifier | None = None,
    prior_text: str | None = None,
) -> pd.DataFrame:
    """Score each statement on the three stance measures AND the descriptive text
    metrics (length, readability, theme intensity, uncertainty, change-magnitude),
    returning one row per statement with exactly HISTORY_COLUMNS.

    - action / lexicon_tone / roberta_stance / n_sentences: as in v1.
    - word_count, flesch, uncertainty_per1k, theme_*: per-statement on clean_statement(text).
    - change_magnitude: word edit-distance vs the prior statement (NaN for the first ever).
    `prior_text` lets an incremental run measure change_magnitude against the statement
    already in history; `statements` is sorted by date before scoring.
    """
    statements = statements.sort_values("date").reset_index(drop=True)
    lexicon_dir = Path(lexicon_dir)
    hawk_a, dove_a = load_lexicon(lexicon_dir / "action_tone.json")
    hawk_l, dove_l = load_lexicon(lexicon_dir / "hawk_dove.json")

    act = score_statements_lexicon(statements, hawk_a, dove_a).rename(columns={"stance": "action"})
    lex = score_statements_lexicon(statements, hawk_l, dove_l).rename(columns={"stance": "lexicon_tone"})
    nsent = pd.DataFrame({
        "date": statements["date"].to_numpy(),
        "n_sentences": [len(split_sentences(t)) for t in statements["text"]],
    })
    metrics = _metric_rows(statements, Path(themes_path), prior_text)

    out = act.merge(lex, on="date").merge(nsent, on="date").merge(metrics, on="date")
    if roberta is not None:
        rob = score_statements(statements, roberta).rename(columns={"stance": "roberta_stance"})
        out = out.merge(rob, on="date", how="left")
    else:
        out["roberta_stance"] = float("nan")
    return out[HISTORY_COLUMNS]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_score.py -v`
Expected: PASS (existing v1 score tests still pass — they call `score_all_measures(..., themes_path=...)`? No: update them — see Step 5).

- [ ] **Step 5: Update the two v1 score tests to pass `themes_path`**

In `tests/test_monitor_score.py`, the v1 tests `test_score_all_measures_columns_and_merge` and `test_score_all_measures_no_roberta_is_nan` call `score_all_measures(_statements(), lexicon_dir="data/lexicons", roberta=...)`. Add `themes_path="data/lexicons/themes.json"` to both calls. Their existing assertions on `action`/`roberta_stance` still hold; the `list(out.columns) == HISTORY_COLUMNS` assertion now includes the new columns automatically (HISTORY_COLUMNS grew).

Run: `python -m pytest tests/test_monitor_score.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/cbp/monitor/score.py tests/test_monitor_score.py
git commit -m "feat(tracker): score_all_measures emits text metrics + change_magnitude"
```

---

### Task 5: Word-level redline

**Files:**
- Modify: `src/cbp/monitor/contrast.py`
- Test: `tests/test_monitor_contrast.py`

- [ ] **Step 1: Write the failing test (append / replace the redline tests)**

```python
# tests/test_monitor_contrast.py — APPEND (the old sentence-level redline assertions
# in test_redline_detects_equal_insert_delete_replace still hold at word granularity,
# but add these word-level + cleaning assertions)
def test_redline_is_word_level_and_strips_boilerplate():
    prev = ("For release at 2:00 p.m. EDT Share The Committee will hold rates steady. "
            "Voting for the monetary policy action were members.")
    curr = ("For release at 2:00 p.m. EDT Share The Committee will raise rates steady. "
            "Voting for the monetary policy action were members.")
    segs = redline(prev, curr)
    # boilerplate removed -> not present in any segment
    joined = " ".join(s["prev"] + s["curr"] for s in segs)
    assert "Voting for" not in joined and "EDT" not in joined
    # exactly the single word hold->raise shows as a replace; the rest equal
    reps = [s for s in segs if s["op"] == "replace"]
    assert len(reps) == 1
    assert "hold" in reps[0]["prev"] and "raise" in reps[0]["curr"]


def test_redline_near_identical_is_mostly_equal():
    prev = "The Committee decided to maintain the target range at current levels."
    curr = "The Committee decided to maintain the target range at current levels today."
    segs = redline(prev, curr)
    assert any(s["op"] == "equal" for s in segs)
    assert any(s["op"] == "insert" and "today" in s["curr"] for s in segs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_contrast.py -k "word_level or near_identical" -v`
Expected: FAIL (boilerplate still present / sentence-level granularity)

- [ ] **Step 3: Rewrite `redline` in `contrast.py`**

Replace the `redline` function and its `split_sentences` import. New top of file + function:

```python
# src/cbp/monitor/contrast.py — replace the import line
#   from cbp.models.stance_scorer import split_sentences
# with:
import re

from cbp.monitor.metrics import clean_statement

_WORDS = re.compile(r"\S+")
```

```python
# src/cbp/monitor/contrast.py — replace the whole redline() function with:
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_contrast.py -v`
Expected: PASS (the word-level tests; the original equal/insert/delete/replace test still passes at word granularity).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/contrast.py tests/test_monitor_contrast.py
git commit -m "feat(tracker): word-level redline over cleaned statement text"
```

---

### Task 6: Wire metrics through the orchestrator

**Files:**
- Modify: `src/cbp/monitor/__main__.py`
- Test: `tests/test_monitor_main.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_monitor_main.py — APPEND
def test_run_monitor_populates_metric_columns(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.calendar_path.write_text("date\n2024-01-31\n2024-03-20\n")
    m.run_monitor(cfg, use_roberta=True, get_html=_fake_get_html, roberta=_fake_roberta)
    hist = load_history(cfg.history_path)
    for c in ["word_count", "flesch", "theme_inflation", "uncertainty_per1k"]:
        assert hist[c].notna().any()
    # second statement gets a change_magnitude vs the first
    assert hist["change_magnitude"].notna().any()
```

Note: `_cfg` builds a `Config(...)`; add `themes_path=__import__("pathlib").Path("data/lexicons/themes.json")` to the `_cfg` helper so the real theme file is used (the test statements contain "raise"/"lower" + theme words from `_HTML`).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_main.py -k "metric_columns" -v`
Expected: FAIL with `TypeError: score_all_measures() missing 1 required keyword-only argument: 'themes_path'`

- [ ] **Step 3: Pass `themes_path` + `prior_text` in `run_monitor`**

In `src/cbp/monitor/__main__.py`, replace the scoring block inside `run_monitor` (the `if not statements.empty:` body) with:

```python
            if not statements.empty:
                clf = roberta if roberta is not None else build_classifier(cfg, use_roberta)
                prior_text = None
                if len(history):
                    last_date = pd.Timestamp(history["date"].iloc[-1]).date()
                    prior = fetch_statements([last_date], cfg.statements_dir, **kw)
                    if not prior.empty:
                        prior_text = prior.iloc[0]["text"]
                scored = score_all_measures(statements, lexicon_dir=cfg.lexicon_dir,
                                            themes_path=cfg.themes_path, roberta=clf,
                                            prior_text=prior_text)
                history = upsert_history(history, scored)
                save_history(history, cfg.history_path)
                _write_latest_redline(cfg, history)
            else:
                logger.warning("no statements fetched for %d pending date(s)", len(todo))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_main.py -v`
Expected: PASS (existing main tests + the new one). The redline JSON written here is now word-level (Task 5).

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/__main__.py tests/test_monitor_main.py
git commit -m "feat(tracker): orchestrator passes themes_path + prior text for change_magnitude"
```

---

### Task 7: Six-panel dashboard — glossary, inline redline, theme heatmap, change-magnitude, comm-style

**Files:**
- Modify: `src/cbp/monitor/site.py`
- Test: `tests/test_monitor_site.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# tests/test_monitor_site.py — APPEND
from cbp.monitor.site import (
    build_theme_heatmap, build_change_magnitude_figure, build_commstyle_figure, glossary_html,
)


def _hist_v2():
    base = _hist()  # the v1 3-row fixture
    base["word_count"] = [300, 320, 280]
    base["flesch"] = [25.0, 22.0, 30.0]
    base["uncertainty_per1k"] = [10.0, 12.0, 8.0]
    base["change_magnitude"] = [float("nan"), 0.2, 0.5]
    for i, c in enumerate(["theme_inflation", "theme_employment", "theme_growth",
                           "theme_balance_sheet", "theme_financial_conditions"]):
        base[c] = [float(i + 1)] * 3
    return base


def test_theme_heatmap_has_five_rows():
    fig = build_theme_heatmap(_hist_v2())
    assert len(fig.data) == 1                       # one heatmap trace
    assert len(fig.data[0].z) == 5                  # five themes on the y axis


def test_change_magnitude_and_commstyle_build():
    assert len(build_change_magnitude_figure(_hist_v2()).data) == 1
    assert len(build_commstyle_figure(_hist_v2()).data) == 3   # words, flesch, uncertainty


def test_glossary_defines_each_measure():
    g = glossary_html()
    for term in ("action", "lexicon", "RoBERTa", "change", "theme"):
        assert term.lower() in g.lower()


def test_redline_html_is_inline_flow():
    segs = [{"op": "equal", "prev": "the rate", "curr": "the rate"},
            {"op": "replace", "prev": "hold", "curr": "raise"}]
    html = build_redline_html(segs)
    assert "redline-flow" in html
    assert "rl-delete" in html and "rl-insert" in html


def test_render_site_has_six_sections(tmp_path):
    out = tmp_path / "index.html"
    deltas = {"date_prior": "2024-03-20", "date_latest": "2024-05-01",
              "action": {"prior": -1.0, "latest": 0.0, "delta": 1.0},
              "lexicon_tone": {"prior": 0.2, "latest": 0.0, "delta": -0.2},
              "roberta_stance": {"prior": 0.1, "latest": None, "delta": None}}
    segs = [{"op": "equal", "prev": "Rates unchanged.", "curr": "Rates unchanged."}]
    render_site(_hist_v2(), deltas, segs, out)
    html = out.read_text(encoding="utf-8")
    for heading in ("How to read", "focused on", "How much", "Communication style"):
        assert heading.lower() in html.lower()
    assert "descriptive" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monitor_site.py -k "heatmap or commstyle or glossary or inline or six_sections" -v`
Expected: FAIL with `ImportError: cannot import name 'build_theme_heatmap'`

- [ ] **Step 3: Rewrite `site.py`**

```python
# src/cbp/monitor/site.py
from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

VERDICT_URL = "https://github.com/alanvaa06/CB_Policy_Analysis/blob/main/docs/results/2026-06-29-lexicon-baseline-verdict.md"
_LEVELS = [("action", "action"), ("lexicon_tone", "lexicon"), ("roberta_stance", "RoBERTa")]
_THEMES = [("theme_inflation", "inflation"), ("theme_employment", "employment"),
           ("theme_growth", "growth"), ("theme_balance_sheet", "balance sheet"),
           ("theme_financial_conditions", "financial")]


def build_levels_figure(history: pd.DataFrame):
    """Stance measures through time (kept, now in the 'stance in context' panel)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col], name=name,
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="Stance measures — levels (1999→latest)",
                      yaxis_title="stance", template="plotly_white", height=360)
    return fig


def build_deltas_figure(history: pd.DataFrame):
    """Meeting-over-meeting Δ per stance measure."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for col, name in _LEVELS:
        fig.add_trace(go.Scatter(x=history["date"], y=history[col].diff(), name=f"Δ {name}",
                                 mode="lines+markers", connectgaps=False))
    fig.update_layout(title="Stance measures — meeting-over-meeting change",
                      yaxis_title="Δ stance", template="plotly_white", height=360)
    return fig


def build_theme_heatmap(history: pd.DataFrame):
    """What the Fed is focused on: theme intensity (hits per 1,000 words) over time."""
    import plotly.graph_objects as go
    z = [history[col].tolist() for col, _ in _THEMES]
    fig = go.Figure(go.Heatmap(z=z, x=history["date"], y=[name for _, name in _THEMES],
                               colorscale="YlOrRd", colorbar={"title": "per 1k"}))
    fig.update_layout(title="What the Fed is focused on — theme intensity (per 1,000 words)",
                      template="plotly_white", height=320)
    return fig


def build_change_magnitude_figure(history: pd.DataFrame):
    """How much each statement was rewritten vs the prior (0=identical, 1=rewritten)."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=history["date"], y=history["change_magnitude"],
                               mode="lines+markers", name="change", connectgaps=False))
    fig.update_layout(title="How much each statement changed vs the prior",
                      yaxis_title="edit fraction (0–1)", template="plotly_white", height=320)
    return fig


def build_commstyle_figure(history: pd.DataFrame):
    """Communication style: length, readability, uncertainty as stacked sub-plots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("length (words)", "readability (Flesch)",
                                        "uncertainty (per 1,000 words)"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["word_count"], name="words"), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=history["flesch"], name="flesch"), row=2, col=1)
    fig.add_trace(go.Scatter(x=history["date"], y=history["uncertainty_per1k"], name="uncertainty"), row=3, col=1)
    fig.update_layout(title="Communication style over time", template="plotly_white",
                      height=520, showlegend=False)
    return fig


def build_redline_html(segments: list[dict]) -> str:
    """Inline word-level redline: one flowing paragraph, only changes highlighted."""
    if not segments:
        return '<p class="rl-empty">Need ≥2 statements to show a redline.</p>'
    parts = []
    for s in segments:
        op = s["op"]
        if op == "equal":
            parts.append(f'<span class="rl-equal">{_html.escape(s["curr"])}</span>')
        elif op == "insert":
            parts.append(f'<span class="rl-insert">{_html.escape(s["curr"])}</span>')
        elif op == "delete":
            parts.append(f'<span class="rl-delete">{_html.escape(s["prev"])}</span>')
        else:  # replace
            parts.append(f'<span class="rl-delete">{_html.escape(s["prev"])}</span> '
                         f'<span class="rl-insert">{_html.escape(s["curr"])}</span>')
    return '<div class="redline-flow">' + " ".join(parts) + "</div>"


def glossary_html() -> str:
    """Plain-English definition of every measure on the page."""
    items = [
        ("action", "the rate decision verb in the statement: +1 hike / 0 hold / −1 cut. Mirrors the decision itself."),
        ("lexicon", "net hawkish−dovish stance words (transparent word-count). Goes silent on 2024+ statements, which drop stance adjectives."),
        ("RoBERTa", "a machine-learning per-sentence stance model, averaged. Populated on the heavy inference run; gapped until then."),
        ("themes", "how often the statement mentions each topic (inflation, employment, growth, balance sheet, financial conditions), per 1,000 words. Presence, not sentiment."),
        ("change magnitude", "how much the wording changed vs the prior statement (0 = identical, 1 = rewritten)."),
        ("communication style", "statement length, Flesch readability, and hedging/uncertainty word density over time."),
    ]
    lis = "".join(f"<li><b>{_html.escape(t)}</b> — {_html.escape(d)}</li>" for t, d in items)
    return ("<p>This is a <b>descriptive tracker</b>: it reads what each FOMC statement "
            "<i>says</i> and <i>how it changed</i> — not a rate forecast.</p>"
            f"<ul class='glossary'>{lis}</ul>")


def _fmt(v) -> str:
    return "—" if v is None else f"{v:+.3f}"


def _deltas_table_html(deltas: dict) -> str:
    if not deltas:
        return '<p class="rl-empty">Need ≥2 statements to compute deltas.</p>'
    rows = []
    for col, name in _LEVELS:
        m = deltas[col]
        rows.append(f"<tr><td>{name}</td><td>{_fmt(m['prior'])}</td>"
                    f"<td>{_fmt(m['latest'])}</td><td><b>{_fmt(m['delta'])}</b></td></tr>")
    return (f"<p>Comparing <b>{deltas['date_latest']}</b> vs <b>{deltas['date_prior']}</b></p>"
            "<table class='deltas'><tr><th>measure</th><th>prior</th><th>latest</th>"
            "<th>Δ</th></tr>" + "".join(rows) + "</table>")


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}
.banner{background:#EAF2FB;border:1px solid #B6D4F2;padding:10px 14px;border-radius:6px;font-size:14px}
h1{font-size:23px}h2{font-size:17px;margin-top:30px;border-bottom:1px solid #eee;padding-bottom:4px}
ul.glossary{font-size:14px;line-height:1.5}ul.glossary li{margin:3px 0}
table.deltas{border-collapse:collapse}table.deltas td,table.deltas th{border:1px solid #ddd;padding:4px 10px;text-align:right}
table.deltas td:first-child,table.deltas th:first-child{text-align:left}
.redline-flow{line-height:1.7;font-size:15px}
.redline-flow span{padding:1px 2px;border-radius:3px}
.rl-equal{color:#444}.rl-insert{background:#E6F4EA;color:#137333}
.rl-delete{background:#FCE8E6;color:#A50E0E;text-decoration:line-through}
.rl-empty{color:#888;font-style:italic}
"""

_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FOMC Statement Tracker</title><style>{css}</style></head><body>
<h1>FOMC Statement Tracker</h1>
<div class="banner">{glossary}</div>
<h2>Latest statement — what changed vs the prior</h2>
{deltas_table}
<div class="redline">{redline}</div>
<h2>What the Fed is focused on</h2>{heatmap}
<h2>How much each statement changed</h2>{change_fig}
<h2>Communication style</h2>{commstyle}
<h2>Stance measures, in context</h2>{levels}{deltas_fig}
</body></html>"""


def render_site(history: pd.DataFrame, deltas: dict, segments: list[dict],
                out_path: Path, *, verdict_url: str = VERDICT_URL) -> None:
    """Assemble the self-contained 6-panel tracker and write it to `out_path`.
    Only the first figure inlines plotly.js; the rest reference it."""
    heatmap = build_theme_heatmap(history).to_html(full_html=False, include_plotlyjs="inline")
    change_fig = build_change_magnitude_figure(history).to_html(full_html=False, include_plotlyjs=False)
    commstyle = build_commstyle_figure(history).to_html(full_html=False, include_plotlyjs=False)
    levels = build_levels_figure(history).to_html(full_html=False, include_plotlyjs=False)
    deltas_fig = build_deltas_figure(history).to_html(full_html=False, include_plotlyjs=False)
    page = _PAGE.format(css=_CSS, glossary=glossary_html(), deltas_table=_deltas_table_html(deltas),
                        redline=build_redline_html(segments), heatmap=heatmap, change_fig=change_fig,
                        commstyle=commstyle, levels=levels, deltas_fig=deltas_fig)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_monitor_site.py -v`
Expected: PASS. (The v1 `test_levels_figure_has_three_traces` etc. still hold — those builders are unchanged in shape.)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/monitor/site.py tests/test_monitor_site.py
git commit -m "feat(tracker): six-panel dashboard — glossary, inline redline, theme heatmap, change-magnitude, comm-style"
```

---

### Task 8: Regenerate the committed dashboard data (torch-free backfill)

**Files:**
- Modify: `data/monitor/tone_history.csv` (regenerated), `data/monitor/latest_redline.json` (regenerated)

- [ ] **Step 1: Delete the v1 history so the run rescores every statement with the new columns**

Run: `rm data/monitor/tone_history.csv`
(The v1 CSV has only the 5 base columns and all 226 dates already "scored" — without deleting it, `pending_dates` is empty and the new metric columns never get computed.)

- [ ] **Step 2: Run the torch-free backfill on the cached statements**

Run: `python -m cbp.monitor --no-roberta`
Expected: logs scoring of the calendar dates (≈226 of 230; the few uncached dates are skipped), then "dashboard written to site\index.html (226 statements)".

- [ ] **Step 3: Verify the regenerated artifacts**

Run:
```bash
python -c "import pandas as pd; d=pd.read_csv('data/monitor/tone_history.csv'); print(len(d),'rows'); print([c for c in d.columns]); print('themes>0:', (d['theme_inflation']>0).sum()); print('chg non-null:', d['change_magnitude'].notna().sum())"
```
Expected: ~226 rows; columns include `word_count, flesch, uncertainty_per1k, change_magnitude, theme_inflation…theme_financial_conditions`; `theme_inflation>0` for most rows; `change_magnitude` non-null for all but the first.

Run: `python -c "import json; s=json.load(open('data/monitor/latest_redline.json'))['segments']; print('segments:', len(s), 'ops:', sorted({x['op'] for x in s}))"`
Expected: a word-level segment list (more, shorter segments than v1; ops include equal/insert/delete/replace).

- [ ] **Step 4: Confirm the suite is still green and the page renders torch-free**

Run: `python -m pytest -q`
Expected: PASS (all).

Run: `python -m cbp.monitor --rebuild-only && test -f site/index.html && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit the data (site/ stays gitignored)**

```bash
git add data/monitor/tone_history.csv data/monitor/latest_redline.json
git commit -m "feat(tracker): regenerate dashboard data with v2 metrics (torch-free; RoBERTa pending)"
```

---

### Task 9: Value-framed README + context docs + full green

**Files:**
- Modify: `README.md`
- Modify: `docs/context/results.md`, `docs/context/todo.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: Replace the README "Statement monitor" section**

Replace the block under `### Statement monitor → dashboard (descriptive, not predictive)` (the v1 fenced commands + the two paragraphs after it) with:

```markdown
### FOMC Statement Tracker (live dashboard)

**Live:** https://alanvaa06.github.io/CB_Policy_Analysis/

A transparent, reproducible reader for every FOMC statement. Each meeting it shows:

- **What changed** — a word-level redline of the latest statement vs the prior one (boilerplate stripped).
- **What the Fed is focused on** — theme intensity over time (inflation, employment, growth, balance sheet, financial conditions).
- **How much it changed** — the edit-distance of each statement vs the one before, so pivotal meetings stand out.
- **Communication style** — statement length, readability, and uncertainty-word density across 1999→today.
- **Stance, in context** — the transparent action/lexicon measures (RoBERTa optional), with an on-page glossary.

```bash
python -m cbp.monitor                 # score new statements + rebuild dashboard (.[site]; add .[infer] for RoBERTa)
python -m cbp.monitor --no-roberta    # torch-free run
python -m cbp.monitor --rebuild-only  # re-render from committed data (the CI path; .[site] only)
```

Each run upserts `data/monitor/tone_history.csv` + `latest_redline.json` (commit both); CI (`.github/workflows/pages.yml`) re-renders torch-free and publishes to `gh-pages`. Extend the meeting list yearly in `data/monitor/fomc_calendar.csv`.
```

- [ ] **Step 2: Update context docs**

```markdown
<!-- docs/context/results.md — prepend -->
- [2026-06-30] Statement Tracker v2 SHIPPED (feat/statement-tracker-v2, PRD 005). Dashboard redesigned: word-level redline over boilerplate-stripped text (fixed the sentence-block wall); new torch-free text metrics (themes per-1k, change-magnitude=1−difflib word-ratio vs prior, length/Flesch/uncertainty) in tone_history.csv; 6 panels incl. glossary + theme heatmap. README value-framed w/ live link. RoBERTa col still pending heavy run.
```
```markdown
<!-- docs/context/todo.md — under open work -->
Tracker v2 (PRD 005) DONE on feat/statement-tracker-v2. Open: [pending] merge → main (CI republishes); [pending] optional heavy `.[infer]` run to fill roberta_stance; [pending] spot-check themes.json stems vs corpus.
```
```markdown
<!-- docs/context/sesion-log.md — append -->
- [2026-06-30]: Dashboard v2 — word-level redline + theme/change-magnitude/comm-style analytics + glossary. Brainstorm→PRD 005→plan→TDD build.
```

- [ ] **Step 3: Full suite green**

Run: `python -m pytest -q`
Expected: PASS (all).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/context/results.md docs/context/todo.md docs/context/sesion-log.md
git commit -m "docs(tracker): value-framed README + live link; context log"
```

---

## Self-review checklist (completed)

- **Spec coverage:** themes.json + load_themes (T1) · clean/word_count/flesch/themes/uncertainty/change_magnitude (T2) · extended schema (T3) · metrics emitted incl. change_magnitude-vs-prior (T4) · word-level redline over cleaned text (T5) · orchestrator wiring + prior_text (T6) · glossary + inline redline + theme heatmap + change-magnitude + comm-style + stance-in-context (T7) · regenerated committed data (T8) · value-framed README + link (T9). All §-sections mapped.
- **Placeholder scan:** none — every code/test step has complete code.
- **Type consistency:** `HISTORY_COLUMNS`/`THEME_COLUMNS`/`METRIC_COLUMNS` defined in T3, imported by T4/T7 tests; `score_all_measures(statements, *, lexicon_dir, themes_path, roberta=None, prior_text=None)` defined T4, called identically in T6; `redline(prev, curr)->list[{op,prev,curr}]` T5 consumed by `build_redline_html` T7; theme dict keys `inflation/employment/growth/balance_sheet/financial_conditions` ↔ `theme_*` columns ↔ `_THEMES` rows are consistent across T1/T3/T4/T7; render path stays torch-free (only `[site]`/plotly), matching the CI contract.
- **Edge cases:** first-row change_magnitude NaN (T4 test), flesch 0.0 on empty (T2 test), clean_statement never empty (T2 test), legacy CSV load tolerated (existing history fill) + regenerated in T8.
